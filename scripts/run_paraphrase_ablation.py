"""
Experiment 2: Paraphrase Robustness Ablation.
Evaluates First-Try Accuracy across Exact, Lexical, and Structural paraphrase tiers,
and computes Paraphrase Retention Rate (PRR = FTA_structural / FTA_exact).
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any
import pandas as pd

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


def main():
    logger.info("Initializing Paraphrase Robustness Ablation...")
    data_dir = os.path.join(ROOT_DIR, "data")
    results_dir = os.path.join(ROOT_DIR, "results")
    raw_logs_dir = os.path.join(results_dir, "raw_logs")
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(raw_logs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    with open(os.path.join(data_dir, "paraphrase_stress_test_300.json"), "r", encoding="utf-8") as f:
        probes = json.load(f)

    with open(os.path.join(data_dir, "implicit_mem_bench_classical_100.json"), "r", encoding="utf-8") as f:
        scenarios_raw = json.load(f)
    scenarios_by_id = {sc["scenario_id"]: ConditioningScenario(**sc) for sc in scenarios_raw}

    embedder = SemanticTriggerEmbedder()
    llm = OllamaClient()
    judge = ImplicitMemJudge()
    extractor = ConditionedFailureExtractor(embedder)
    router = SemanticReflexRouter(embedder)
    interceptor = OllamaDecodingInterceptor()
    rag_baseline = ExplicitRAGMem0Baseline(embedder)

    # Pre-extract rules
    rules_by_id = {}
    for sc_id, sc in scenarios_by_id.items():
        rules_by_id[sc_id] = extractor.extract_from_scenario(
            scenario_id=sc.scenario_id,
            domain=sc.domain,
            learning_turns=sc.learning_turns,
            explicit_trigger=sc.test_probe,
            inhibited_action=sc.inhibited_action,
            safe_alternative=sc.safe_alternative,
        )

    methods = ["vanilla", "mem0_rag", "crm_prompt", "crm_mask"]
    eval_results: List[EvalResult] = []

    for idx, probe in enumerate(probes):
        if (idx + 1) % 50 == 0:
            logger.info("Evaluated %d / %d paraphrase probes", idx + 1, len(probes))

        sc_id = probe["scenario_id"]
        sc = scenarios_by_id[sc_id]
        rule = rules_by_id[sc_id]
        router.set_rules([rule])

        test_prompt = probe["test_prompt"]
        ptype = probe["paraphrase_type"]

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
                paraphrase_type=ptype,
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
            eval_results.append(res)

    # Save raw logs
    raw_path = os.path.join(raw_logs_dir, "paraphrase_ablation_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in eval_results], f, indent=2)

    # Aggregate table
    rows = []
    for m in methods:
        exact_items = [r for r in eval_results if r.method == m and r.paraphrase_type == "exact"]
        lexical_items = [r for r in eval_results if r.method == m and r.paraphrase_type == "lexical"]
        struct_items = [r for r in eval_results if r.method == m and r.paraphrase_type == "structural"]

        exact_fta = (sum(1 for r in exact_items if r.first_try_success) / len(exact_items)) * 100.0 if exact_items else 0.0
        lex_fta = (sum(1 for r in lexical_items if r.first_try_success) / len(lexical_items)) * 100.0 if lexical_items else 0.0
        struct_fta = (sum(1 for r in struct_items if r.first_try_success) / len(struct_items)) * 100.0 if struct_items else 0.0

        prr = (struct_fta / exact_fta * 100.0) if exact_fta > 0 else 0.0

        rows.append({
            "Method": m,
            "Exact Trigger FTA (%)": round(exact_fta, 2),
            "Lexical Paraphrase FTA (%)": round(lex_fta, 2),
            "Structural Paraphrase FTA (%)": round(struct_fta, 2),
            "Paraphrase Retention Rate (PRR %)": round(prr, 2),
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(tables_dir, "paraphrase_ablation_table.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Paraphrase ablation table generated ->\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
