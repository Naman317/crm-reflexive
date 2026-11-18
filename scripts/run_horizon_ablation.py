"""
Experiment 3: Interference Horizon Scaling Ablation.
Evaluates First-Try Accuracy as a function of distraction turns M in {4, 12, 30, 120}
(~200, ~500, ~2000, and ~8000 tokens) on the Negative-Association Inhibition subset.
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

# Additional pool of realistic distraction turns to scale horizon up to M=120
EXTENDED_DISTRACTOR_POOL = [
    ("How does binary search work?", "Binary search repeatedly divides a sorted array in half to find target items in O(log n) time."),
    ("What is an IP address?", "An IP address is a unique numerical label assigned to each device connected to a computer network."),
    ("Name three sorting algorithms.", "Quick sort, merge sort, and heap sort."),
    ("What is the function of the ribosome?", "Ribosomes synthesize proteins by translating messenger RNA."),
    ("Calculate the square root of 144.", "The square root of 144 is 12."),
    ("Define polymorphism in OOP.", "Polymorphism is the ability of different classes to respond to the same method invocation with custom behaviors."),
    ("What causes ocean tides?", "Ocean tides are caused by gravitational attraction between the earth, moon, and sun."),
    ("What is the boiling point of water?", "Water boils at 100°C (212°F) at standard atmospheric pressure."),
    ("What is an SSD?", "An SSD (solid state drive) is a persistent flash storage medium offering high read/write throughput."),
    ("Convert 1 kilometer to miles.", "1 kilometer is approximately 0.621371 miles.")
]


def create_extended_interference_turns(target_turns: int) -> List[Dict[str, str]]:
    turns = []
    for i in range(target_turns):
        pair = EXTENDED_DISTRACTOR_POOL[i % len(EXTENDED_DISTRACTOR_POOL)]
        turns.append({"role": "user", "content": f"[Turn {i+1}] {pair[0]}"})
        turns.append({"role": "assistant", "content": pair[1]})
    return turns


def main():
    logger.info("Initializing Interference Horizon Scaling Ablation...")
    data_dir = os.path.join(ROOT_DIR, "data")
    results_dir = os.path.join(ROOT_DIR, "results")
    raw_logs_dir = os.path.join(results_dir, "raw_logs")
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(raw_logs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    with open(os.path.join(data_dir, "implicit_mem_bench_classical_100.json"), "r", encoding="utf-8") as f:
        classical_items = json.load(f)

    # Filter to Negative-Association (Inhibition) subset (50 items)
    inh_items = [ConditioningScenario(**sc) for sc in classical_items if sc["association_type"] == "inhibition"]
    logger.info("Evaluating on %d Negative-Association Inhibition scenarios...", len(inh_items))

    embedder = SemanticTriggerEmbedder()
    llm = OllamaClient()
    judge = ImplicitMemJudge()
    extractor = ConditionedFailureExtractor(embedder)
    router = SemanticReflexRouter(embedder)
    interceptor = OllamaDecodingInterceptor()
    rag_baseline = ExplicitRAGMem0Baseline(embedder)

    rules_by_id = {}
    for sc in inh_items:
        rules_by_id[sc.scenario_id] = extractor.extract_from_scenario(
            scenario_id=sc.scenario_id,
            domain=sc.domain,
            learning_turns=sc.learning_turns,
            explicit_trigger=sc.test_probe,
            inhibited_action=sc.inhibited_action,
            safe_alternative=sc.safe_alternative,
        )

    horizons = [4, 12, 30, 120]  # token equivalents: ~200, ~500, ~2000, ~8000
    methods = ["vanilla", "mem0_rag", "crm_prompt", "crm_mask"]

    eval_results: List[EvalResult] = []

    for M in horizons:
        logger.info("--- Testing Horizon M = %d turns ---", M)
        for sc in inh_items:
            # Clone scenario with extended interference turns
            sc_m = ConditioningScenario(
                scenario_id=sc.scenario_id,
                scenario_type=sc.scenario_type,
                association_type=sc.association_type,
                domain=sc.domain,
                learning_turns=sc.learning_turns,
                interference_turns=create_extended_interference_turns(M),
                test_probe=sc.test_probe,
                paraphrases=sc.paraphrases,
                inhibited_action=sc.inhibited_action,
                safe_alternative=sc.safe_alternative,
                ground_truth_target=sc.ground_truth_target,
            )

            rule = rules_by_id[sc.scenario_id]
            router.set_rules([rule])
            test_prompt = sc_m.test_probe

            for method in methods:
                t0 = time.perf_counter()
                response_text = ""
                router_score = 0.0
                gate_activated = False

                if method == "vanilla":
                    prompt = VanillaBaseline.format_prompt(sc_m, test_prompt)
                    response_text = llm.generate(prompt)

                elif method == "mem0_rag":
                    prompt = rag_baseline.format_prompt(sc_m, test_prompt)
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
                inh_p, safe_p, success = judge.evaluate_response(sc_m, response_text)

                res = EvalResult(
                    scenario_id=sc_m.scenario_id,
                    model_name=llm.active_model,
                    method=method,
                    scenario_type=sc_m.scenario_type,
                    association_type=sc_m.association_type,
                    domain=sc_m.domain,
                    paraphrase_type="canonical",
                    test_prompt=test_prompt,
                    generated_response=response_text,
                    router_score=router_score,
                    gate_activated=gate_activated,
                    inhibited_action_present=inh_p,
                    safe_alternative_present=safe_p,
                    first_try_success=success,
                    interference_horizon=M,
                    latency_ms=latency_ms,
                )
                eval_results.append(res)

    # Save raw logs
    raw_path = os.path.join(raw_logs_dir, "horizon_ablation_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in eval_results], f, indent=2)

    # Aggregate table
    rows = []
    for m in methods:
        row = {"Method": m}
        for M in horizons:
            sub = [r for r in eval_results if r.method == m and r.interference_horizon == M]
            fta = (sum(1 for r in sub if r.first_try_success) / len(sub)) * 100.0 if sub else 0.0
            row[f"M={M} turns (~{M*65} tok)"] = round(fta, 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(tables_dir, "horizon_ablation_table.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Interference horizon ablation table generated ->\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
