"""
GPU Experiment Execution Harness for Colab / Kaggle.
Runs matched five-method comparison on Transformers backend:
- vanilla
- mem0_rag
- crm_prompt
- crm_mask (real HF LogitsProcessor)
- crm_mask_redirect (HF LogitsProcessor + prefix steering)
Supports Qwen2.5-3B and Qwen2.5-7B (4-bit bitsandbytes).
Fully resumable, streams each trial to JSONL immediately.
"""

import os
import sys
import json
import time
import random
import argparse
from typing import Dict, Any, List

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import load_config
from src.embedder import SemanticTriggerEmbedder
from src.router import SemanticRouter
from src.hf_client import HFTransformersClient
from src.scoring import score_trial, extract_action_variants
from src.methods.vanilla import VanillaMethod
from src.methods.mem0_rag import Mem0RAGMethod
from src.methods.crm_prompt import CRMPromptMethod
from src.methods.crm_mask import CRMMaskMethod
from src.methods.crm_mask_redirect import CRMMaskRedirectMethod


def load_done_set(jsonl_path: str) -> set:
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        m = obj.get("method")
                        scen = obj.get("scenario_id")
                        cond = f"{obj.get('paraphrase_type')}_M{obj.get('interference_horizon')}"
                        s = obj.get("seed")
                        done.add((m, scen, cond, s))
                    except Exception:
                        pass
    return done


def run_gpu_suite(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    load_in_4bit: bool = False,
    horizons: List[int] = [4, 12, 30, 120],
    seeds: List[int] = [42, 123, 999],
    max_trials: int = None,
    shard_idx: int = 1,
    shard_total: int = 1,
    output_dir: str = None,
):
    config = load_config()
    out_dir = output_dir or os.path.join(root_dir, config.paths.raw_logs_dir)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "gpu_experiments.jsonl")

    print("=" * 70)
    print("GPU EXPERIMENT EXECUTION SUITE")
    print(f"Model: {model_name} | 4-bit: {load_in_4bit} | Shard: {shard_idx}/{shard_total}")
    print(f"Output: {jsonl_path}")
    print("=" * 70)

    # 1. Load Data
    test_path = os.path.join(root_dir, "data", "test_tasks_160.json")
    with open(test_path, "r", encoding="utf-8") as f:
        all_test_tasks = json.load(f)

    distractor_path = os.path.join(root_dir, "data", "distractor_pool_150.json")
    with open(distractor_path, "r", encoding="utf-8") as f:
        distractor_pool = json.load(f)

    paraphrase_path = os.path.join(root_dir, "data", "paraphrase_stress_test_300.json")
    with open(paraphrase_path, "r", encoding="utf-8") as f:
        paraphrase_probes = json.load(f)

    # 2. Initialize HF client and router
    try:
        hf_client = HFTransformersClient(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            max_new_tokens=48,
        )
    except Exception as e:
        print(f"ERROR: Could not load {model_name}: {e}")
        # Record failure metadata
        meta_err_path = os.path.join(out_dir, "model_load_error.json")
        with open(meta_err_path, "w") as f:
            json.dump({"model": model_name, "error": str(e)}, f, indent=2)
        return

    embedder = SemanticTriggerEmbedder(model_name=config.embedder.model_name, device="cuda" if hf_client.device == "cuda" else "cpu")
    router = SemanticRouter(embedder=embedder, default_tau=config.parameters.tau_default)

    vanilla = VanillaMethod(llm_client=hf_client)
    mem0_rag = Mem0RAGMethod(llm_client=hf_client, embedder=embedder, top_k=config.parameters.rag_top_k)
    crm_prompt = CRMPromptMethod(llm_client=hf_client, router=router)
    crm_mask = CRMMaskMethod(llm_client=hf_client, router=router, beta=config.parameters.mask_beta)
    crm_redirect = CRMMaskRedirectMethod(llm_client=hf_client, router=router, beta=config.parameters.mask_beta)

    method_map = {
        "vanilla": vanilla,
        "mem0_rag": mem0_rag,
        "crm_prompt": crm_prompt,
        "crm_mask": crm_mask,
        "crm_mask_redirect": crm_redirect,
    }

    # Index all test tasks
    router.clear()
    mem0_rag.clear()
    for t in all_test_tasks:
        trigger = t.get("learning_prompt", t.get("test_probe", ""))
        router.add_rule(
            scenario_id=t.get("scenario_id", ""),
            trigger_prompt=trigger,
            a_inhibit=t.get("inhibited_action", ""),
            a_safe=t.get("safe_alternative", ""),
            domain=t.get("domain", ""),
        )
        mem0_rag.index_task(t)

    done_set = load_done_set(jsonl_path)
    print(f"Loaded {len(done_set)} previously completed trials.")

    # Build evaluation plan
    plan = []
    # Horizon runs: 160 tasks x M in horizons x seeds x 5 methods
    for h in horizons:
        distractors_h = distractor_pool[:h]
        for s in seeds:
            # Vary distractor order per seed
            rng = random.Random(s)
            distractors_shuffled = list(distractors_h)
            rng.shuffle(distractors_shuffled)

            for t in all_test_tasks:
                scen_id = t["scenario_id"]
                cond = f"canonical_M{h}"
                for m_name in method_map:
                    if (m_name, scen_id, cond, s) not in done_set:
                        plan.append({
                            "method": m_name,
                            "task": t,
                            "distractors": distractors_shuffled,
                            "test_prompt": t.get("test_probe", ""),
                            "paraphrase_type": "canonical",
                            "horizon": h,
                            "seed": s,
                        })

    # Shard plan
    if shard_total > 1:
        plan = [p for i, p in enumerate(plan) if (i % shard_total) == (shard_idx - 1)]

    if max_trials:
        plan = plan[:max_trials]

    total_trials = len(plan)
    print(f"Total trials to execute in this run: {total_trials}")

    if total_trials == 0:
        print("All trials already complete!")
        return

    with open(jsonl_path, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(plan, 1):
            m_name = item["method"]
            m_obj = method_map[m_name]
            t = item["task"]
            h = item["horizon"]
            s = item["seed"]
            probe = item["test_prompt"]
            tier = item["paraphrase_type"]

            t0 = time.perf_counter()
            if m_name in ["crm_prompt", "crm_mask", "crm_mask_redirect"]:
                res = m_obj.run_trial(task=t, distractors=item["distractors"], test_prompt=probe, tau=config.parameters.tau_default, seed=s)
            else:
                res = m_obj.run_trial(task=t, distractors=item["distractors"], test_prompt=probe, seed=s)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            resp = res.get("generated_response", "")

            scoring = score_trial(
                association_type=t.get("association_type", "inhibition"),
                generated_response=resp,
                a_inhibit=t.get("inhibited_action", ""),
                a_safe=t.get("safe_alternative", ""),
            )

            record = {
                "scenario_id": t["scenario_id"],
                "model_name": model_name,
                "backend": "transformers",
                "method": m_name,
                "scenario_type": t.get("scenario_type", "classical"),
                "association_type": t.get("association_type", "inhibition"),
                "domain": t.get("domain", ""),
                "paraphrase_type": tier,
                "test_prompt": probe,
                "final_prompt_sent": res.get("final_prompt_sent", ""),
                "generated_response": resp,
                "router_score": res.get("router_score", 0.0),
                "tau": config.parameters.tau_default,
                "gate_activated": res.get("gate_activated", False),
                "inhibited_action_present": scoring["inhibited_action_present"],
                "safe_alternative_present": scoring["safe_alternative_present"],
                "first_try_success": scoring["first_try_success"],
                "strict_safe": scoring["strict_safe"],
                "interference_horizon": h,
                "prompt_tokens": res.get("prompt_tokens", 0),
                "latency_ms": elapsed_ms,
                "seed": s,
                "error_message": res.get("error_message"),
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if idx % 20 == 0 or idx == total_trials:
                print(f"[{idx:4d}/{total_trials}] {m_name:<18s} | M={h:3d} | s={s} | tokens={record['prompt_tokens']:4d} | FTA={scoring['first_try_success']} ({elapsed_ms:.1f}ms)")

    print(f"\nExecution complete! Appended {total_trials} rows to {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--4bit", action="store_true", dest="load_in_4bit")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--shard", type=str, default="1/1")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    s_idx, s_tot = map(int, args.shard.split("/"))
    run_gpu_suite(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        max_trials=args.max_trials,
        shard_idx=s_idx,
        shard_total=s_tot,
        output_dir=args.output_dir,
    )
