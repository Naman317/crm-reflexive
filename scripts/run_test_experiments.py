"""
Test Set Live Evaluation Suite.
Runs Experiments A, B, and C against live local Ollama (qwen2.5:3b) on the frozen TEST partition:
- Experiment A: 160 TEST tasks x 4 methods, horizon M=8.
- Experiment B: 240 Paraphrase probes (80 classical TEST x {exact, lexical, structural}) x 4 methods, M=8.
- Experiment C: 80 classical TEST tasks x 4 methods x horizons M in [4, 12, 30, 120].
Continuously checkpoints each trial to results/raw_logs/*.json.
"""

import os
import sys
import json
import time
import random
import logging
from typing import List, Dict, Any, Tuple

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import load_config
from src.embedder import SemanticTriggerEmbedder
from src.router import SemanticRouter
from src.llm_client import LiveOllamaClient
from src.scoring import score_trial
from src.methods.vanilla import VanillaMethod
from src.methods.mem0_rag import Mem0RAGMethod
from src.methods.crm_prompt import CRMPromptMethod
from src.methods.crm_mask import CRMMaskMethod
from checks import run_full_anti_stub_audit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def execute_trial_record(
    method_name: str,
    method_obj: Any,
    task: Dict[str, Any],
    distractors: List[Dict[str, str]],
    test_prompt: str,
    paraphrase_type: str,
    horizon: int,
    tau: float,
    seed: int,
    model_name: str,
) -> Dict[str, Any]:
    scen_id = task.get("scenario_id", "")
    assoc_type = task.get("association_type", "inhibition")
    a_inhibit = task.get("inhibited_action", "")
    a_safe = task.get("safe_alternative", "")
    domain = task.get("domain", "")

    if method_name in ["crm_prompt", "crm_mask"]:
        res = method_obj.run_trial(
            task=task,
            distractors=distractors,
            test_prompt=test_prompt,
            tau=tau,
            seed=seed,
        )
    else:
        res = method_obj.run_trial(
            task=task,
            distractors=distractors,
            test_prompt=test_prompt,
            seed=seed,
        )

    gen_resp = res.get("generated_response", "")
    scoring = score_trial(
        association_type=assoc_type,
        generated_response=gen_resp,
        a_inhibit=a_inhibit,
        a_safe=a_safe,
    )

    return {
        "scenario_id": scen_id,
        "model_name": model_name,
        "backend": res.get("backend", "ollama"),
        "method": method_name,
        "scenario_type": task.get("scenario_type", "classical"),
        "association_type": assoc_type,
        "domain": domain,
        "paraphrase_type": paraphrase_type,
        "test_prompt": test_prompt,
        "final_prompt_sent": res.get("final_prompt_sent", ""),
        "generated_response": gen_resp,
        "router_score": res.get("router_score", 0.0),
        "tau": res.get("tau", tau),
        "gate_activated": res.get("gate_activated", False),
        "inhibited_action_present": scoring["inhibited_action_present"],
        "safe_alternative_present": scoring["safe_alternative_present"],
        "first_try_success": scoring["first_try_success"],
        "strict_safe": scoring["strict_safe"],
        "interference_horizon": horizon,
        "prompt_tokens": res.get("prompt_tokens", 0),
        "latency_ms": res.get("latency_ms", 0.0),
        "seed": seed,
        "error_message": res.get("error_message"),
    }


def run_all_test_experiments(
    exp_targets: List[str] = ["A", "B", "C"],
    seeds: List[int] = None,
    max_tasks_override: int = None,
    max_c_tasks: int = 30,
):
    config = load_config()
    seeds = seeds or config.model.seeds
    out_dir = os.path.join(root_dir, config.paths.raw_logs_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Datasets
    test_path = os.path.join(root_dir, "data", "test_tasks_160.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_tasks = json.load(f)

    if max_tasks_override:
        test_tasks = test_tasks[:max_tasks_override]

    classical_test = [t for t in test_tasks if t.get("scenario_type") == "classical"]
    procedural_test = [t for t in test_tasks if t.get("scenario_type") == "procedural"]

    paraphrase_path = os.path.join(root_dir, "data", "paraphrase_stress_test_300.json")
    with open(paraphrase_path, "r", encoding="utf-8") as f:
        all_paraphrase = json.load(f)

    classical_test_ids = {t["scenario_id"] for t in classical_test}
    test_paraphrase = [p for p in all_paraphrase if p["scenario_id"] in classical_test_ids]

    distractors_path = os.path.join(root_dir, "data", "distractor_pool_150.json")
    with open(distractors_path, "r", encoding="utf-8") as f:
        distractor_pool = json.load(f)

    print("=" * 70)
    print("INITIALIZING TEST EVALUATION HARNESS (160 TEST TASKS)")
    print(f"Classical Test: {len(classical_test)} | Procedural Test: {len(procedural_test)}")
    print(f"Test Paraphrase Probes: {len(test_paraphrase)} | Distractor Pool: {len(distractor_pool)}")
    print("=" * 70)

    # 2. Setup Models and Methods
    embedder = SemanticTriggerEmbedder(model_name=config.embedder.model_name, device=config.embedder.device)
    router = SemanticRouter(embedder=embedder, default_tau=config.parameters.tau_default)
    llm = LiveOllamaClient(
        endpoint=config.model.ollama_endpoint,
        model=config.model.name,
        temperature=config.model.temperature,
        num_predict=48,  # Compact prediction for tool action
        timeout_seconds=config.model.timeout_seconds,
    )

    vanilla = VanillaMethod(llm_client=llm)
    mem0_rag = Mem0RAGMethod(llm_client=llm, embedder=embedder, top_k=config.parameters.rag_top_k)
    crm_prompt = CRMPromptMethod(llm_client=llm, router=router)
    crm_mask = CRMMaskMethod(llm_client=llm, router=router, beta=config.parameters.mask_beta)

    methods = [
        ("vanilla", vanilla),
        ("mem0_rag", mem0_rag),
        ("crm_prompt", crm_prompt),
        ("crm_mask", crm_mask),
    ]

    # Index all TEST tasks in router and mem0
    router.clear()
    mem0_rag.clear()
    for t in test_tasks:
        trigger = t.get("learning_prompt", t.get("test_probe", ""))
        router.add_rule(
            scenario_id=t.get("scenario_id", ""),
            trigger_prompt=trigger,
            a_inhibit=t.get("inhibited_action", ""),
            a_safe=t.get("safe_alternative", ""),
            domain=t.get("domain", ""),
        )
        mem0_rag.index_task(t)

    # =========================================================================
    # EXPERIMENT A: MAIN BENCHMARK (160 tasks x 4 methods, M=8)
    # =========================================================================
    if "A" in exp_targets:
        exp_a_path = os.path.join(out_dir, "main_benchmark_results.json")
        records_a = []
        if os.path.exists(exp_a_path):
            try:
                with open(exp_a_path, "r", encoding="utf-8") as f:
                    records_a = json.load(f)
            except Exception:
                records_a = []

        completed_a = {(r.get("scenario_id"), r.get("method"), r.get("seed", 42)) for r in records_a}
        print(f"\n[EXPERIMENT A] Main Benchmark on {len(test_tasks)} TEST tasks (M=8, Seeds: {seeds})")
        print(f"Already completed: {len(completed_a)} trials.")

        for seed in seeds:
            # Vary distractor ordering deterministically by seed
            rng = random.Random(seed)
            shuffled_distractors = list(distractor_pool)
            rng.shuffle(shuffled_distractors)
            distractors_8 = shuffled_distractors[:8]

            for idx, task in enumerate(test_tasks, 1):
                scen_id = task["scenario_id"]
                test_prompt = task.get("test_probe", "")

                for m_name, m_obj in methods:
                    if (scen_id, m_name, seed) in completed_a:
                        continue

                    rec = execute_trial_record(
                        method_name=m_name,
                        method_obj=m_obj,
                        task=task,
                        distractors=distractors_8,
                        test_prompt=test_prompt,
                        paraphrase_type="canonical",
                        horizon=8,
                        tau=config.parameters.tau_default,
                        seed=seed,
                        model_name=config.model.name,
                    )
                    records_a.append(rec)
                    completed_a.add((scen_id, m_name, seed))

                    # Continuous incremental checkpointing
                    with open(exp_a_path, "w", encoding="utf-8") as f:
                        json.dump(records_a, f, indent=2)

                    if len(records_a) % 10 == 0:
                        print(f"  [Exp A: {len(records_a)} trials] Saved. Latest: {scen_id} | {m_name} (FTA={rec['first_try_success']})")

        print(f"[Done Exp A] Saved {len(records_a)} trials to {exp_a_path}")

    # =========================================================================
    # EXPERIMENT B: PARAPHRASE ABLATION (TEST classical x {exact, lexical, structural}, M=8)
    # =========================================================================
    if "B" in exp_targets:
        exp_b_path = os.path.join(out_dir, "paraphrase_ablation_results.json")
        records_b = []
        if os.path.exists(exp_b_path):
            try:
                with open(exp_b_path, "r", encoding="utf-8") as f:
                    records_b = json.load(f)
            except Exception:
                records_b = []

        completed_b = {(r.get("scenario_id"), r.get("paraphrase_type"), r.get("method"), r.get("seed", 42)) for r in records_b}
        print(f"\n[EXPERIMENT B] Paraphrase Robustness on {len(test_paraphrase)} probes (M=8, Seeds: {seeds})")
        print(f"Already completed: {len(completed_b)} trials.")

        # Map scenario_id to task metadata
        task_map = {t["scenario_id"]: t for t in test_tasks}

        for seed in seeds:
            rng = random.Random(seed)
            shuffled_distractors = list(distractor_pool)
            rng.shuffle(shuffled_distractors)
            distractors_8 = shuffled_distractors[:8]

            for idx, probe in enumerate(test_paraphrase, 1):
                scen_id = probe["scenario_id"]
                p_type = probe["paraphrase_type"]
                test_prompt = probe["test_prompt"]
                parent_task = task_map.get(scen_id, probe)

                for m_name, m_obj in methods:
                    if (scen_id, p_type, m_name, seed) in completed_b:
                        continue

                    rec = execute_trial_record(
                        method_name=m_name,
                        method_obj=m_obj,
                        task=parent_task,
                        distractors=distractors_8,
                        test_prompt=test_prompt,
                        paraphrase_type=p_type,
                        horizon=8,
                        tau=config.parameters.tau_default,
                        seed=seed,
                        model_name=config.model.name,
                    )
                    records_b.append(rec)
                    completed_b.add((scen_id, p_type, m_name, seed))

                    with open(exp_b_path, "w", encoding="utf-8") as f:
                        json.dump(records_b, f, indent=2)

                    if len(records_b) % 10 == 0:
                        print(f"  [Exp B: {len(records_b)} trials] Saved. Latest: {scen_id} [{p_type}] | {m_name}")

        print(f"[Done Exp B] Saved {len(records_b)} trials to {exp_b_path}")

    # =========================================================================
    # EXPERIMENT C: HORIZON ABLATION (Classical TEST x 4 methods x M in [4, 12, 30, 120])
    # =========================================================================
    if "C" in exp_targets:
        exp_c_path = os.path.join(out_dir, "horizon_ablation_results.json")
        records_c = []
        if os.path.exists(exp_c_path):
            try:
                with open(exp_c_path, "r", encoding="utf-8") as f:
                    records_c = json.load(f)
            except Exception:
                records_c = []

        completed_c = {(r.get("scenario_id"), r.get("interference_horizon"), r.get("method"), r.get("seed", 42)) for r in records_c}
        horizons = config.parameters.horizons
        eval_classical = classical_test[:max_c_tasks] if max_c_tasks else classical_test
        print(f"\n[EXPERIMENT C] Horizon Interference Scaling ({len(eval_classical)} tasks x {horizons} horizons x {len(seeds)} seeds)")
        print(f"Already completed: {len(completed_c)} trials.")

        for seed in seeds:
            rng = random.Random(seed)
            shuffled_distractors = list(distractor_pool)
            rng.shuffle(shuffled_distractors)

            for M in horizons:
                distractors_M = shuffled_distractors[:M]

                for idx, task in enumerate(eval_classical, 1):
                    scen_id = task["scenario_id"]
                    test_prompt = task.get("test_probe", "")

                    for m_name, m_obj in methods:
                        if (scen_id, M, m_name, seed) in completed_c:
                            continue

                        rec = execute_trial_record(
                            method_name=m_name,
                            method_obj=m_obj,
                            task=task,
                            distractors=distractors_M,
                            test_prompt=test_prompt,
                            paraphrase_type="canonical",
                            horizon=M,
                            tau=config.parameters.tau_default,
                            seed=seed,
                            model_name=config.model.name,
                        )
                        records_c.append(rec)
                        completed_c.add((scen_id, M, m_name, seed))

                        with open(exp_c_path, "w", encoding="utf-8") as f:
                            json.dump(records_c, f, indent=2)

                        if len(records_c) % 10 == 0:
                            print(f"  [Exp C: {len(records_c)} trials] Saved. Latest: M={M} | {scen_id} | {m_name} (tokens={rec['prompt_tokens']})")

        print(f"[Done Exp C] Saved {len(records_c)} trials to {exp_c_path}")

    # =========================================================================
    # ANTI-STUB AUDIT GATING
    # =========================================================================
    all_active_records = []
    if "A" in exp_targets:
        all_active_records.extend(records_a)
    if "B" in exp_targets:
        all_active_records.extend(records_b)
    if "C" in exp_targets:
        all_active_records.extend(records_c)

    audit_pass = run_full_anti_stub_audit(all_active_records, src_root=os.path.join(root_dir, "src"), check_horizon=("C" in exp_targets))
    if not audit_pass:
        print("[CRITICAL] Anti-stub audit FAILED! Exiting non-zero.")
        sys.exit(1)
    print("\n[COMPLETE] All live benchmark evaluations finished and verified cleanly!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run test experiments suite")
    parser.add_argument("--targets", nargs="+", default=["A", "B", "C"], help="Target experiments to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to evaluate")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max tasks override for Exp A")
    parser.add_argument("--max-c-tasks", type=int, default=30, help="Max classical tasks for Exp C")
    args = parser.parse_args()

    run_all_test_experiments(
        exp_targets=args.targets,
        seeds=args.seeds,
        max_tasks_override=args.max_tasks,
        max_c_tasks=args.max_c_tasks,
    )
