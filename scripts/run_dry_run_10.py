"""
Dry run execution script on 10 classical tasks against LIVE local Ollama LLM.
Validates:
- Real model generations for all 4 methods (vanilla, mem0_rag, crm_prompt, crm_mask).
- Scoring engine output.
- Anti-stub checks passing.
- Outputs 10 raw transcripts per method for user inspection before full run.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

# Ensure project root is in sys.path
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


def run_dry_run(limit: int = 10, seed: int = 42, horizon: int = 4) -> List[Dict[str, Any]]:
    config = load_config()
    print("=" * 70)
    print(f"STARTING LIVE-LLM DRY RUN ON {limit} TASKS")
    print(f"Model: {config.model.name} via {config.model.ollama_endpoint}")
    print(f"Temperature: {config.model.temperature} | Seed: {seed} | Horizon: {horizon}")
    print("=" * 70)

    # 1. Initialize components
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

    # 2. Load data
    tasks_path = os.path.join(root_dir, config.paths.data_dir, "implicit_mem_bench_classical_100.json")
    distractors_path = os.path.join(root_dir, config.paths.data_dir, "distractor_pool_150.json")

    with open(tasks_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)
    tasks = all_tasks[:limit]

    with open(distractors_path, "r", encoding="utf-8") as f:
        distractor_pool = json.load(f)

    # Slice distractors for horizon
    distractors = distractor_pool[:horizon]

    # 3. Register learning phase rules into router & mem0
    router.clear()
    mem0_rag.clear()

    for t in tasks:
        trigger = t.get("learning_prompt", t.get("test_probe", ""))
        a_inhibit = t.get("inhibited_action", "")
        a_safe = t.get("safe_alternative", "")
        domain = t.get("domain", "")
        scen_id = t.get("scenario_id", "")

        router.add_rule(
            scenario_id=scen_id,
            trigger_prompt=trigger,
            a_inhibit=a_inhibit,
            a_safe=a_safe,
            domain=domain,
        )
        mem0_rag.index_task(t)

    # 4. Run trials across all 4 methods
    methods = [
        ("vanilla", vanilla),
        ("mem0_rag", mem0_rag),
        ("crm_prompt", crm_prompt),
        ("crm_mask", crm_mask),
    ]

    records = []
    total_evals = len(tasks) * len(methods)
    count = 0

    for task_idx, task in enumerate(tasks, 1):
        scen_id = task.get("scenario_id")
        test_prompt = task.get("test_probe", "")
        assoc_type = task.get("association_type", "inhibition")
        a_inhibit = task.get("inhibited_action", "")
        a_safe = task.get("safe_alternative", "")
        domain = task.get("domain", "")

        print(f"\n--- Task {task_idx}/{len(tasks)}: [{scen_id}] ({assoc_type}) ---")
        print(f"Trigger: {test_prompt}")
        print(f"Forbidden: {a_inhibit} | Safe: {a_safe}")

        for method_name, method_obj in methods:
            count += 1
            print(f"  -> Running {method_name:<10s} ... ", end="", flush=True)

            t0 = time.perf_counter()
            if method_name in ["crm_prompt", "crm_mask"]:
                trial_res = method_obj.run_trial(
                    task=task,
                    distractors=distractors,
                    test_prompt=test_prompt,
                    tau=config.parameters.tau_default,
                    seed=seed,
                )
            else:
                trial_res = method_obj.run_trial(
                    task=task,
                    distractors=distractors,
                    test_prompt=test_prompt,
                    seed=seed,
                )

            # Score trial
            gen_resp = trial_res.get("generated_response", "")
            scoring_res = score_trial(
                association_type=assoc_type,
                generated_response=gen_resp,
                a_inhibit=a_inhibit,
                a_safe=a_safe,
            )

            record = {
                "scenario_id": scen_id,
                "model_name": config.model.name,
                "backend": trial_res.get("backend", "ollama"),
                "method": method_name,
                "scenario_type": task.get("scenario_type", "classical"),
                "association_type": assoc_type,
                "domain": domain,
                "paraphrase_type": "canonical",
                "test_prompt": test_prompt,
                "final_prompt_sent": trial_res.get("final_prompt_sent", ""),
                "generated_response": gen_resp,
                "router_score": trial_res.get("router_score", 0.0),
                "tau": trial_res.get("tau", config.parameters.tau_default),
                "gate_activated": trial_res.get("gate_activated", False),
                "inhibited_action_present": scoring_res["inhibited_action_present"],
                "safe_alternative_present": scoring_res["safe_alternative_present"],
                "first_try_success": scoring_res["first_try_success"],
                "strict_safe": scoring_res["strict_safe"],
                "interference_horizon": horizon,
                "prompt_tokens": trial_res.get("prompt_tokens", 0),
                "latency_ms": trial_res.get("latency_ms", 0.0),
                "seed": seed,
                "error_message": trial_res.get("error_message"),
            }
            records.append(record)

            succ_str = "SUCCESS" if record["first_try_success"] else "FAIL"
            print(f"Done in {record['latency_ms']:.1f}ms | Gate={record['gate_activated']} | {succ_str}")

    # 5. Run anti-stub audit
    audit_ok = run_full_anti_stub_audit(records, src_root=os.path.join(root_dir, "src"))
    if not audit_ok:
        print("[ERROR] Anti-stub audit failed on dry run! Exiting non-zero.")
        sys.exit(1)

    # 6. Save dry run records
    out_dir = os.path.join(root_dir, config.paths.raw_logs_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "dry_run_10_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved {len(records)} dry run records to {out_file}")

    return records


if __name__ == "__main__":
    records = run_dry_run(limit=10, seed=42, horizon=4)
