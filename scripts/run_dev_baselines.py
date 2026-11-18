"""
Generate 10 matched live transcripts for Vanilla, Mem0 RAG, and CRM-Mask on the 10 DEV tasks.
"""
import os
import sys
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import load_config
from src.llm_client import LiveOllamaClient
from src.embedder import SemanticTriggerEmbedder
from src.router import SemanticRouter
from src.methods.vanilla import VanillaMethod
from src.methods.mem0_rag import Mem0RAGMethod
from src.methods.crm_mask import CRMMaskMethod
from src.scoring import score_trial


def run_baselines():
    config = load_config()
    llm = LiveOllamaClient(
        endpoint=config.model.ollama_endpoint,
        model=config.model.name,
        temperature=config.model.temperature,
        num_predict=64,
        timeout_seconds=config.model.timeout_seconds,
    )
    embedder = SemanticTriggerEmbedder()
    router = SemanticRouter(embedder=embedder)

    dev_path = os.path.join(root_dir, "data", "dev_tasks_40.json")
    distractors_path = os.path.join(root_dir, "data", "distractor_pool_150.json")

    with open(dev_path, "r", encoding="utf-8") as f:
        dev_tasks = json.load(f)[:10]
    with open(distractors_path, "r", encoding="utf-8") as f:
        distractor_pool = json.load(f)[:4]

    # Initialize methods
    vanilla = VanillaMethod(llm_client=llm)
    mem0 = Mem0RAGMethod(llm_client=llm, embedder=embedder, top_k=3)
    for t in dev_tasks:
        mem0.index_task(t)
        router.add_rule(
            scenario_id=t["scenario_id"],
            trigger_prompt=t.get("learning_prompt", t.get("test_probe", "")),
            a_inhibit=t.get("inhibited_action", ""),
            a_safe=t.get("safe_alternative", ""),
            domain=t.get("domain", ""),
        )

    crm_mask = CRMMaskMethod(llm_client=llm, router=router, beta=20.0)

    results = {"vanilla": [], "mem0_rag": [], "crm_mask": []}

    print("Running 10 DEV tasks for Vanilla...")
    for idx, t in enumerate(dev_tasks, 1):
        test_prompt = t.get("test_probe", "")
        res = vanilla.run_trial(task=t, distractors=distractor_pool, test_prompt=test_prompt, seed=42)
        resp = res.get("generated_response", "")
        sc = score_trial(t.get("association_type", "inhibition"), resp, t.get("inhibited_action", ""), t.get("safe_alternative", ""))
        res["scoring"] = sc
        res["scenario_id"] = t["scenario_id"]
        res["test_prompt"] = test_prompt
        results["vanilla"].append(res)
        print(f"  [Vanilla {idx}/10] {t['scenario_id']}: FTA={sc['first_try_success']}")

    print("\nRunning 10 DEV tasks for Mem0 RAG...")
    for idx, t in enumerate(dev_tasks, 1):
        test_prompt = t.get("test_probe", "")
        res = mem0.run_trial(task=t, distractors=distractor_pool, test_prompt=test_prompt, seed=42)
        resp = res.get("generated_response", "")
        sc = score_trial(t.get("association_type", "inhibition"), resp, t.get("inhibited_action", ""), t.get("safe_alternative", ""))
        res["scoring"] = sc
        res["scenario_id"] = t["scenario_id"]
        res["test_prompt"] = test_prompt
        results["mem0_rag"].append(res)
        print(f"  [Mem0 {idx}/10] {t['scenario_id']}: FTA={sc['first_try_success']} (Inh={sc['inhibited_action_present']})")

    print("\nRunning 10 DEV tasks for CRM-Mask (Stop Interception)...")
    for idx, t in enumerate(dev_tasks, 1):
        test_prompt = t.get("test_probe", "")
        res = crm_mask.run_trial(task=t, distractors=distractor_pool, test_prompt=test_prompt, tau=0.35, seed=42)
        resp = res.get("generated_response", "")
        sc = score_trial(t.get("association_type", "inhibition"), resp, t.get("inhibited_action", ""), t.get("safe_alternative", ""))
        res["scoring"] = sc
        res["scenario_id"] = t["scenario_id"]
        res["test_prompt"] = test_prompt
        results["crm_mask"].append(res)
        print(f"  [CRM-Mask {idx}/10] {t['scenario_id']}: FTA={sc['first_try_success']} (Inh={sc['inhibited_action_present']})")

    out_path = os.path.join(root_dir, "results", "dev_baselines_10.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baselines to {out_path}")


if __name__ == "__main__":
    run_baselines()
