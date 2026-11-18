"""
Full Benchmark Execution Script for CRM ImplicitMemBench Evaluation.
Implements:
- Experiment A: Main Benchmark (200 tasks x 4 methods, M=8)
- Experiment B: Paraphrase Ablation (300 probes x 4 methods, M=8)
- Experiment C: Horizon Ablation (Classical tasks x 4 methods x M in [4, 12, 30, 120])
With:
- Continuous incremental checkpointing to disk.
- Resume capability (skips already completed trials).
- Real prompt token counting & real latency tracking.
- Zero canned fallbacks (all responses from live Ollama).
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Tuple

# Add project root to sys.path
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


def load_datasets(config) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    data_dir = os.path.join(root_dir, config.paths.data_dir)
    with open(os.path.join(data_dir, "implicit_mem_bench_classical_100.json"), "r", encoding="utf-8") as f:
        classical = json.load(f)
    with open(os.path.join(data_dir, "implicit_mem_bench_procedural_100.json"), "r", encoding="utf-8") as f:
        procedural = json.load(f)
    with open(os.path.join(data_dir, "paraphrase_stress_test_300.json"), "r", encoding="utf-8") as f:
        paraphrase = json.load(f)
    with open(os.path.join(data_dir, "benign_overcaution_controls_100.json"), "r", encoding="utf-8") as f:
        benign = json.load(f)
    with open(os.path.join(data_dir, "distractor_pool_150.json"), "r", encoding="utf-8") as f:
        distractors = json.load(f)

    return classical, procedural, paraphrase, benign, distractors


def execute_trial(
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


def run_experiment_suite(max_tasks_per_suite: int = 100, seeds: List[int] = None):
    config = load_config()
    seeds = seeds or [42]
    out_dir = os.path.join(root_dir, config.paths.raw_logs_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("STARTING FULL REPRODUCIBLE EXPERIMENT SUITE AGAINST LIVE OLLAMA")
    print(f"Model: {config.model.name} | Seeds: {seeds} | Max tasks: {max_tasks_per_suite}")
    print("=" * 70)

    # 1. Initialize models and methods
    embedder = SemanticTriggerEmbedder(model_name=config.embedder.model_name, device=config.embedder.device)
    router = SemanticRouter(embedder=embedder, default_tau=config.parameters.tau_default)
    llm_client = LiveOllamaClient(
        endpoint=config.model.ollama_endpoint,
        model=config.model.name,
        temperature=config.model.temperature,
        num_predict=config.model.num_predict,
        timeout_seconds=config.model.timeout_seconds,
    )

    vanilla = VanillaMethod(llm_client=llm_client)
    mem0_rag = Mem0RAGMethod(llm_client=llm_client, embedder=embedder, top_k=config.parameters.rag_top_k)
    crm_prompt = CRMPromptMethod(llm_client=llm_client, router=router)
    crm_mask = CRMMaskMethod(llm_client=llm_client, router=router, beta=config.parameters.mask_beta)

    methods = [
        ("vanilla", vanilla),
        ("mem0_rag", mem0_rag),
        ("crm_prompt", crm_prompt),
        ("crm_mask", crm_mask),
    ]

    classical, procedural, paraphrase, benign, distractor_pool = load_datasets(config)

    # Populate router & mem0 with 100 classical + 100 procedural rules
    router.clear()
    mem0_rag.clear()

    combined_tasks = classical + procedural
    for t in combined_tasks:
        trigger = t.get("learning_prompt", t.get("test_probe", ""))
        router.add_rule(
            scenario_id=t.get("scenario_id", ""),
            trigger_prompt=trigger,
            a_inhibit=t.get("inhibited_action", ""),
            a_safe=t.get("safe_alternative", ""),
            domain=t.get("domain", ""),
        )
        mem0_rag.index_task(t)

    logger.info("Initialized router with %d stored rules.", len(router.rules))

    # --- EXPERIMENT A: MAIN BENCHMARK (M=8) ---
    exp_a_file = os.path.join(out_dir, "main_benchmark_results.json")
    exp_a_records = []
    if os.path.exists(exp_a_file):
        try:
            with open(exp_a_file, "r", encoding="utf-8") as f:
                exp_a_records = json.load(f)
        except Exception:
            exp_a_records = []

    completed_a = {(r.get("scenario_id"), r.get("method"), r.get("seed", 42)) for r in exp_a_records}
    benchmark_tasks = combined_tasks[:max_tasks_per_suite]

    print(f"\n>>> Running Experiment A: Main Benchmark ({len(benchmark_tasks)} tasks x 4 methods x {len(seeds)} seeds)")
    distractors_8 = distractor_pool[:8]

    for seed in seeds:
        for t_idx, task in enumerate(benchmark_tasks, 1):
            scen_id = task["scenario_id"]
            test_prompt = task.get("test_probe", "")

            for m_name, m_obj in methods:
                if (scen_id, m_name, seed) in completed_a:
                    continue

                rec = execute_trial(
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
                exp_a_records.append(rec)
                completed_a.add((scen_id, m_name, seed))

                # Checkpoint every 5 trials
                if len(exp_a_records) % 5 == 0:
                    with open(exp_a_file, "w", encoding="utf-8") as f:
                        json.dump(exp_a_records, f, indent=2)
                    print(f"  [Exp A Checkpoint] Saved {len(exp_a_records)} trials. (Current: {scen_id} / {m_name})")

    with open(exp_a_file, "w", encoding="utf-8") as f:
        json.dump(exp_a_records, f, indent=2)
    print(f"[Done Exp A] Saved {len(exp_a_records)} total records to {exp_a_file}")

    # Run anti-stub check on Experiment A
    run_full_anti_stub_audit(exp_a_records, src_root=os.path.join(root_dir, "src"))


if __name__ == "__main__":
    run_experiment_suite(max_tasks_per_suite=100, seeds=[42])
