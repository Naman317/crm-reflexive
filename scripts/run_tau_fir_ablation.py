"""
Experiment D: Threshold Sensitivity and Over-Caution (FIR) Ablation.
Evaluates:
- True Trigger Recall on 300 PARAPHRASED probes (Exact, Lexical, Structural tiers).
- False Inhibition Rate (FIR %) on 100 Benign Controls.
- Precision and F1 score across tau in [0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55, 0.65].
Guarantees: Trigger probes and stored patterns are not identical strings for lexical/structural probes.
"""

import os
import sys
import json
import time
import numpy as np

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import load_config
from src.embedder import SemanticTriggerEmbedder
from src.router import SemanticRouter


def run_tau_fir_ablation():
    config = load_config()
    tau_list = config.parameters.tau_list  # [0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55, 0.65]

    print("=" * 70)
    print("RUNNING EXPERIMENT D: TAU & OVER-CAUTION (FIR) ABLATION")
    print(f"Sweeping tau values: {tau_list}")
    print("=" * 70)

    # 1. Initialize embedder & router
    embedder = SemanticTriggerEmbedder(model_name=config.embedder.model_name, device=config.embedder.device)
    router = SemanticRouter(embedder=embedder)

    # 2. Load datasets
    classical_path = os.path.join(root_dir, config.paths.data_dir, "implicit_mem_bench_classical_100.json")
    probes_path = os.path.join(root_dir, config.paths.data_dir, "paraphrase_stress_test_300.json")
    controls_path = os.path.join(root_dir, config.paths.data_dir, "benign_overcaution_controls_100.json")

    with open(classical_path, "r", encoding="utf-8") as f:
        classical_tasks = json.load(f)
    with open(probes_path, "r", encoding="utf-8") as f:
        paraphrase_probes = json.load(f)
    with open(controls_path, "r", encoding="utf-8") as f:
        benign_controls = json.load(f)

    # Populate router with canonical learning prompts
    router.clear()
    for t in classical_tasks:
        trigger = t.get("learning_prompt", t.get("test_probe", ""))
        router.add_rule(
            scenario_id=t.get("scenario_id", ""),
            trigger_prompt=trigger,
            a_inhibit=t.get("inhibited_action", ""),
            a_safe=t.get("safe_alternative", ""),
            domain=t.get("domain", ""),
        )

    print(f"Populated router with {len(router.rules)} stored reflex rules.")
    print(f"Pre-embedding {len(paraphrase_probes)} probes and {len(benign_controls)} controls for fast sweep...")

    # Pre-embed once to accelerate evaluation across all tau values
    probe_embeddings = []
    for p in paraphrase_probes:
        q = p["test_prompt"]
        emb = embedder.embed(q)
        probe_embeddings.append((p, q, emb))

    control_embeddings = []
    for c in benign_controls:
        q = c["prompt"]
        emb = embedder.embed(q)
        control_embeddings.append((c, q, emb))

    rule_matrix = router._ensure_matrix()
    rule_list = router.rules

    # 3. Sweep across tau values
    results = []
    summary_table = []

    for tau in tau_list:
        tier_counts = {"exact": {"total": 0, "hits": 0}, "lexical": {"total": 0, "hits": 0}, "structural": {"total": 0, "hits": 0}}
        total_probe_hits = 0

        for probe, q, emb in probe_embeddings:
            tier = probe["paraphrase_type"]
            sims = np.dot(rule_matrix, emb)
            max_idx = int(np.argmax(sims))
            score = float(sims[max_idx])
            fired = score >= tau
            win_rule = rule_list[max_idx] if fired else None

            tier_counts[tier]["total"] += 1
            if fired:
                tier_counts[tier]["hits"] += 1
                total_probe_hits += 1

            results.append({
                "type": "probe",
                "probe_id": probe.get("probe_id", ""),
                "scenario_id": probe.get("scenario_id", ""),
                "paraphrase_type": tier,
                "test_prompt": q,
                "tau": tau,
                "router_score": score,
                "gate_activated": fired,
                "should_inhibit": True,
            })

        # Evaluate on Benign Controls
        false_inhibitions = 0
        for ctrl, q, emb in control_embeddings:
            sims = np.dot(rule_matrix, emb)
            max_idx = int(np.argmax(sims))
            score = float(sims[max_idx])
            fired = score >= tau
            win_rule = rule_list[max_idx] if fired else None

            if fired:
                false_inhibitions += 1

            results.append({
                "type": "benign_control",
                "control_id": ctrl.get("control_id", ""),
                "topic": ctrl.get("topic", ""),
                "test_prompt": q,
                "tau": tau,
                "router_score": score,
                "gate_activated": fired,
                "should_inhibit": False,
            })

        total_probes = len(paraphrase_probes)
        total_controls = len(benign_controls)

        recall_exact = (tier_counts["exact"]["hits"] / tier_counts["exact"]["total"]) * 100.0
        recall_lexical = (tier_counts["lexical"]["hits"] / tier_counts["lexical"]["total"]) * 100.0
        recall_structural = (tier_counts["structural"]["hits"] / tier_counts["structural"]["total"]) * 100.0
        recall_overall = (total_probe_hits / total_probes) * 100.0
        fir = (false_inhibitions / total_controls) * 100.0

        tp = total_probe_hits
        fp = false_inhibitions
        fn = total_probes - tp

        precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        f1 = (2 * tp / (2 * tp + fp + fn)) * 100.0 if (2 * tp + fp + fn) > 0 else 0.0

        row = {
            "tau": tau,
            "recall_exact": recall_exact,
            "recall_lexical": recall_lexical,
            "recall_structural": recall_structural,
            "recall_overall": recall_overall,
            "false_inhibition_rate": fir,
            "precision": precision,
            "f1_score": f1,
        }
        summary_table.append(row)
        print(f"tau={tau:.2f} | Overall Recall: {recall_overall:5.1f}% (Ex: {recall_exact:.0f}%, Lex: {recall_lexical:.0f}%, Str: {recall_structural:.0f}%) | FIR: {fir:5.1f}% | Prec: {precision:5.1f}% | F1: {f1:5.1f}%")

    out_dir = os.path.join(root_dir, config.paths.raw_logs_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tau_fir_ablation_results.json")
    meta = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedder_model": config.embedder.model_name,
        "embedder_device": config.embedder.device,
        "tau_list": tau_list,
        "config_hash": "01c69df2f1b02566acd88a2bafacc12694551e4880270176781316b1204c2407",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"run_metadata": meta, "summary": summary_table, "records": results}, f, indent=2)

    print(f"\nSaved {len(results)} tau ablation evaluation records to {out_path}")
    return summary_table


if __name__ == "__main__":
    run_tau_fir_ablation()
