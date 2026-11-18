"""
Verification of M=120 Horizon fix with num_ctx >= 10240.
Times trials, prints prompt_tokens, latency, and response.
"""

import sys
import os
import time
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)

from src.config import load_config
from src.llm_client import LiveOllamaClient
from src.embedder import SemanticTriggerEmbedder
from src.router import SemanticRouter
from src.methods.vanilla import VanillaMethod
from src.methods.mem0_rag import Mem0RAGMethod
from src.methods.crm_prompt import CRMPromptMethod
from src.methods.crm_mask import CRMMaskMethod
from scripts.run_test_experiments import execute_trial_record


def run_sample():
    config = load_config()
    print("=" * 70)
    print("M=120 HORIZON VERIFICATION SUITE")
    print(f"Model: {config.model.name} | num_ctx: {config.model.num_ctx} | timeout: {config.model.timeout_seconds}s")
    print("=" * 70)

    llm = LiveOllamaClient(
        endpoint=config.model.ollama_endpoint,
        model=config.model.name,
        temperature=config.model.temperature,
        num_predict=48,
        num_ctx=config.model.num_ctx,
        timeout_seconds=config.model.timeout_seconds,
    )

    embedder = SemanticTriggerEmbedder(model_name=config.embedder.model_name, device=config.embedder.device)
    router = SemanticRouter(embedder=embedder, default_tau=config.parameters.tau_default)
    mem0_rag = Mem0RAGMethod(llm_client=llm, embedder=embedder, top_k=config.parameters.rag_top_k)
    vanilla = VanillaMethod(llm_client=llm)
    crm_prompt = CRMPromptMethod(llm_client=llm, router=router)
    crm_mask = CRMMaskMethod(llm_client=llm, router=router, beta=config.parameters.mask_beta)

    with open(os.path.join(root_dir, "data", "test_tasks_160.json"), "r", encoding="utf-8") as f:
        tasks = json.load(f)

    with open(os.path.join(root_dir, "data", "distractor_pool_150.json"), "r", encoding="utf-8") as f:
        pool = json.load(f)
    distractors_120 = pool[:120]

    # Index tasks
    for t in tasks[:10]:
        trigger = t.get("learning_prompt", t.get("test_probe", ""))
        router.add_rule(
            scenario_id=t.get("scenario_id", ""),
            trigger_prompt=trigger,
            a_inhibit=t.get("inhibited_action", ""),
            a_safe=t.get("safe_alternative", ""),
            domain=t.get("domain", ""),
        )
        mem0_rag.index_task(t)

    # 10 test evaluations across diverse methods and tasks
    sample_plan = [
        ("vanilla", vanilla, tasks[0]),
        ("crm_prompt", crm_prompt, tasks[0]),
        ("crm_mask", crm_mask, tasks[0]),
        ("mem0_rag", mem0_rag, tasks[0]),
        ("crm_prompt", crm_prompt, tasks[1]),
        ("vanilla", vanilla, tasks[1]),
        ("crm_mask", crm_mask, tasks[1]),
        ("mem0_rag", mem0_rag, tasks[1]),
        ("crm_prompt", crm_prompt, tasks[2]),
        ("vanilla", vanilla, tasks[2]),
    ]

    print(f"\nRunning 10-row live evaluation at M=120...")
    records = []

    for idx, (m_name, m_obj, task) in enumerate(sample_plan, 1):
        scen_id = task["scenario_id"]
        t_start = time.perf_counter()
        rec = execute_trial_record(
            method_name=m_name,
            method_obj=m_obj,
            task=task,
            distractors=distractors_120,
            test_prompt=task.get("test_probe", ""),
            paraphrase_type="canonical",
            horizon=120,
            tau=0.35,
            seed=42,
            model_name=config.model.name,
        )
        dur = time.perf_counter() - t_start
        records.append(rec)

        print(
            f"[{idx:2d}/10] M=120 | {scen_id[:28]:28s} | {m_name:10s} | "
            f"tokens={rec['prompt_tokens']:4d} | latency={dur:5.1f}s | "
            f"gate={str(rec['gate_activated']):5s} | safe={str(rec['safe_alternative_present']):5s} | "
            f"err={rec['error_message']}"
        )

    out_path = os.path.join(root_dir, "results", "raw_logs", "m120_sample_10.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved 10-row sample to {out_path}")
    return records


if __name__ == "__main__":
    run_sample()
