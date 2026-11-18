"""
Experiment 1: Main Benchmark Execution.
Evaluates Vanilla LLM, Mem0-style Explicit RAG, CRM-Prompt, and CRM-Mask
across Procedural Memory, Classical Conditioning (Overall, Inhibition, Preference).
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any
import pandas as pd

# Add repo root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.schemas import ConditioningScenario, EvalResult, ReflexRule
from src.embedder import SemanticTriggerEmbedder
from src.extractor import ConditionedFailureExtractor
from src.router import SemanticReflexRouter
from src.inhibitory_gate import CRMPromptGate, OllamaDecodingInterceptor
from src.baselines import VanillaBaseline, ExplicitRAGMem0Baseline
from src.llm_backends import OllamaClient
from src.judge import ImplicitMemJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(path: str) -> List[ConditioningScenario]:
    with open(path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    scenarios = []
    for item in raw_items:
        scenarios.append(ConditioningScenario(**item))
    return scenarios


def run_evaluation(
    scenarios: List[ConditioningScenario],
    methods: List[str],
    llm: OllamaClient,
    embedder: SemanticTriggerEmbedder,
    judge: ImplicitMemJudge,
    tau: float = 0.35,
) -> List[EvalResult]:
    extractor = ConditionedFailureExtractor(embedder)
    router = SemanticReflexRouter(embedder, tau=tau)
    interceptor = OllamaDecodingInterceptor()
    rag_baseline = ExplicitRAGMem0Baseline(embedder)

    # Pre-extract reflex rules for all scenarios into the reflex buffer
    reflex_rules: Dict[str, ReflexRule] = {}
    for sc in scenarios:
        r = extractor.extract_from_scenario(
            scenario_id=sc.scenario_id,
            domain=sc.domain,
            learning_turns=sc.learning_turns,
            explicit_trigger=sc.test_probe,
            inhibited_action=sc.inhibited_action,
            safe_alternative=sc.safe_alternative,
        )
        reflex_rules[sc.scenario_id] = r

    results: List[EvalResult] = []

    for idx, sc in enumerate(scenarios):
        if (idx + 1) % 25 == 0:
            logger.info("Processed %d / %d scenarios", idx + 1, len(scenarios))

        rule = reflex_rules[sc.scenario_id]
        router.set_rules([rule])

        test_prompt = sc.test_probe

        for method in methods:
            t0 = time.perf_counter()
            response_text = ""
            router_score = 0.0
            gate_activated = False

            if method == "vanilla":
                prompt = VanillaBaseline.format_prompt(sc, test_prompt)
                response_text = llm.generate(prompt)

            elif method == "mem0_rag":
                prompt = rag_baseline.format_prompt(sc, test_prompt)
                response_text = llm.generate(prompt)

            elif method == "crm_prompt":
                gate_activated, winning_rule, router_score = router.route(test_prompt)
                if gate_activated and winning_rule:
                    prompt = CRMPromptGate.construct_prompt(test_prompt, winning_rule)
                else:
                    prompt = test_prompt
                response_text = llm.generate(prompt)

            elif method == "crm_mask":
                gate_activated, winning_rule, router_score = router.route(test_prompt)
                options = {}
                if gate_activated and winning_rule:
                    options = interceptor.prepare_ollama_options(winning_rule)
                prompt = test_prompt
                raw_response = llm.generate(prompt, options=options)
                if gate_activated and winning_rule:
                    response_text, _ = interceptor.intercept_and_filter(raw_response, winning_rule)
                else:
                    response_text = raw_response

            latency_ms = (time.perf_counter() - t0) * 1000.0
            inh_p, safe_p, success = judge.evaluate_response(sc, response_text)

            res = EvalResult(
                scenario_id=sc.scenario_id,
                model_name=llm.active_model,
                method=method,
                scenario_type=sc.scenario_type,
                association_type=sc.association_type,
                domain=sc.domain,
                paraphrase_type="canonical",
                test_prompt=test_prompt,
                generated_response=response_text,
                router_score=router_score,
                gate_activated=gate_activated,
                inhibited_action_present=inh_p,
                safe_alternative_present=safe_p,
                first_try_success=success,
                interference_horizon=len(sc.interference_turns),
                latency_ms=latency_ms,
            )
            results.append(res)

    return results


def main():
    logger.info("Initializing Main Benchmark Evaluation...")
    data_dir = os.path.join(ROOT_DIR, "data")
    results_dir = os.path.join(ROOT_DIR, "results")
    raw_logs_dir = os.path.join(results_dir, "raw_logs")
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(raw_logs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    classical_scenarios = load_dataset(os.path.join(data_dir, "implicit_mem_bench_classical_100.json"))
    procedural_scenarios = load_dataset(os.path.join(data_dir, "implicit_mem_bench_procedural_100.json"))

    embedder = SemanticTriggerEmbedder()
    llm = OllamaClient()
    judge = ImplicitMemJudge()

    methods = ["vanilla", "mem0_rag", "crm_prompt", "crm_mask"]

    logger.info("Evaluating Classical Conditioning Suite (100 items)...")
    res_classical = run_evaluation(classical_scenarios, methods, llm, embedder, judge)

    logger.info("Evaluating Procedural Memory Suite (100 items)...")
    res_procedural = run_evaluation(procedural_scenarios, methods, llm, embedder, judge)

    all_results = res_classical + res_procedural

    # Save raw logs
    raw_path = os.path.join(raw_logs_dir, "main_benchmark_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    logger.info("Saved raw logs -> %s", raw_path)

    # Compute aggregate metrics
    records = []
    for m in methods:
        # Classical Overall
        c_items = [r for r in res_classical if r.method == m]
        c_fta = sum(1 for r in c_items if r.first_try_success) / len(c_items) * 100.0

        # Classical Inhibition (Negative)
        inh_items = [r for r in res_classical if r.method == m and r.association_type == "inhibition"]
        inh_fta = sum(1 for r in inh_items if r.first_try_success) / len(inh_items) * 100.0

        # Classical Preference (Positive)
        pref_items = [r for r in res_classical if r.method == m and r.association_type == "preference"]
        pref_fta = sum(1 for r in pref_items if r.first_try_success) / len(pref_items) * 100.0

        # Procedural Memory
        p_items = [r for r in res_procedural if r.method == m]
        p_fta = sum(1 for r in p_items if r.first_try_success) / len(p_items) * 100.0

        records.append({
            "Method": m,
            "Procedural Memory (FTA %)": round(p_fta, 2),
            "Classical Conditioning (FTA %)": round(c_fta, 2),
            "Inhibition Subset (FTA %)": round(inh_fta, 2),
            "Preference Subset (FTA %)": round(pref_fta, 2),
            "Asymmetry Gap (Pref - Inh)": round(pref_fta - inh_fta, 2),
        })

    # Add Published ACL 2026 Reference Baselines (Qin et al., 2026 Table 2 / Table 5)
    published_baselines = [
        {"Method": "DeepSeek-R1 (Published ACL'26)", "Procedural Memory (FTA %)": 58.00, "Classical Conditioning (FTA %)": 69.67, "Inhibition Subset (FTA %)": 42.00, "Preference Subset (FTA %)": 88.00, "Asymmetry Gap (Pref - Inh)": 46.00},
        {"Method": "Qwen3-32B (Published ACL'26)", "Procedural Memory (FTA %)": 55.00, "Classical Conditioning (FTA %)": 67.00, "Inhibition Subset (FTA %)": 40.50, "Preference Subset (FTA %)": 86.50, "Asymmetry Gap (Pref - Inh)": 46.00},
        {"Method": "GPT-5 (Published ACL'26)", "Procedural Memory (FTA %)": 52.00, "Classical Conditioning (FTA %)": 64.00, "Inhibition Subset (FTA %)": 38.00, "Preference Subset (FTA %)": 85.00, "Asymmetry Gap (Pref - Inh)": 47.00},
        {"Method": "Qwen3-8B (Published ACL'26)", "Procedural Memory (FTA %)": 51.00, "Classical Conditioning (FTA %)": 64.00, "Inhibition Subset (FTA %)": 37.00, "Preference Subset (FTA %)": 84.00, "Asymmetry Gap (Pref - Inh)": 47.00},
        {"Method": "Claude-4.1-opus (Published ACL'26)", "Procedural Memory (FTA %)": 35.00, "Classical Conditioning (FTA %)": 41.67, "Inhibition Subset (FTA %)": 22.00, "Preference Subset (FTA %)": 76.00, "Asymmetry Gap (Pref - Inh)": 54.00},
        {"Method": "LLaMA-3.1-8B-Instruct (Published ACL'26)", "Procedural Memory (FTA %)": 32.00, "Classical Conditioning (FTA %)": 38.33, "Inhibition Subset (FTA %)": 19.50, "Preference Subset (FTA %)": 74.00, "Asymmetry Gap (Pref - Inh)": 54.50},
        {"Method": "Qwen2.5-7B-Instruct (Published ACL'26)", "Procedural Memory (FTA %)": 28.00, "Classical Conditioning (FTA %)": 35.67, "Inhibition Subset (FTA %)": 17.60, "Preference Subset (FTA %)": 75.00, "Asymmetry Gap (Pref - Inh)": 57.40},
    ]

    all_table_records = records + published_baselines
    df = pd.DataFrame(all_table_records)
    csv_path = os.path.join(tables_dir, "main_benchmark_table.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Main benchmark table generated ->\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
