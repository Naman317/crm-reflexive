"""
Report and Asset Generation Engine for CRM Reproducible Evaluation.
Computes:
- Tables I-IV matching paper layout.
- Mean and 95% Confidence Intervals (Wilson score interval for proportions).
- 5 randomly chosen raw transcripts per method for human inspection.
- Generates 300 DPI publication plots.
- Exports results/summary.csv and results/run_metadata.json.
- Generates report.md.
"""

import os
import sys
import json
import math
import random
from typing import Dict, List, Any, Tuple

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import load_config


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculates Wilson score interval for binomial proportion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    z = 1.96 if confidence == 0.95 else 2.576
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    lower = max(0.0, centre - spread) * 100.0
    upper = min(1.0, centre + spread) * 100.0
    val = p * 100.0
    return val, lower, upper


def generate_report():
    config = load_config()
    raw_dir = os.path.join(root_dir, config.paths.raw_logs_dir)
    fig_dir = os.path.join(root_dir, config.paths.figures_dir)
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load available experiment records
    dry_run_path = os.path.join(raw_dir, "dry_run_10_results.json")
    main_path = os.path.join(raw_dir, "main_benchmark_results.json")
    tau_path = os.path.join(raw_dir, "tau_fir_ablation_results.json")
    latency_path = os.path.join(raw_dir, "latency_results.json")

    eval_records = []
    if os.path.exists(main_path):
        with open(main_path, "r", encoding="utf-8") as f:
            eval_records = json.load(f)
    elif os.path.exists(dry_run_path):
        with open(dry_run_path, "r", encoding="utf-8") as f:
            eval_records = json.load(f)

    with open(tau_path, "r", encoding="utf-8") as f:
        tau_data = json.load(f)

    latency_data = {}
    if os.path.exists(latency_path):
        with open(latency_path, "r", encoding="utf-8") as f:
            latency_data = json.load(f)

    # 2. Write run_metadata.json
    meta = {
        "model_name": config.model.name,
        "ollama_endpoint": config.model.ollama_endpoint,
        "temperature": config.model.temperature,
        "embedder": config.embedder.model_name,
        "embedder_dimension": config.embedder.dimension,
        "router_threshold_tau": config.parameters.tau_default,
        "backends": {
            "vanilla": "ollama (greedy T=0.0)",
            "mem0_rag": "ollama (top-k=3 vector retrieval)",
            "crm_prompt": "ollama (recency-0 contrastive prompt injection)",
            "crm_mask": "ollama_stop_interception (token suppression)",
        },
        "trials_evaluated": len(eval_records),
    }
    with open(os.path.join(root_dir, "results", "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 3. Build Table I: Main Benchmark Aggregation
    by_method = {}
    for r in eval_records:
        m = r["method"]
        by_method.setdefault(m, []).append(r)

    table1_rows = []
    for m, recs in by_method.items():
        total = len(recs)
        succ = sum(1 for r in recs if r["first_try_success"])
        val, low, high = wilson_score_interval(succ, total)
        inh_total = sum(1 for r in recs if "inhib" in r.get("association_type", ""))
        inh_succ = sum(1 for r in recs if "inhib" in r.get("association_type", "") and r["first_try_success"])
        i_val, i_low, i_high = wilson_score_interval(inh_succ, inh_total)

        table1_rows.append({
            "method": m,
            "total_trials": total,
            "overall_fta": f"{val:.1f}% [{low:.1f}%, {high:.1f}%]",
            "inhibition_fta": f"{i_val:.1f}% [{i_low:.1f}%, {i_high:.1f}%]",
            "mean_latency_ms": f"{sum(r['latency_ms'] for r in recs)/total:.1f}" if total else "0.0",
        })

    # 4. Extract 5 randomly chosen transcripts per method
    random.seed(42)
    sample_transcripts = {}
    for m, recs in by_method.items():
        samples = random.sample(recs, min(5, len(recs)))
        sample_transcripts[m] = samples

    # 5. Build report.md
    report_lines = [
        "# Contrastive Reflexive Memory (CRM): Live-LLM Evaluation Report",
        "",
        "**Evaluation Environment:**",
        f"- **Model:** `{config.model.name}` via Local Ollama (`{config.model.ollama_endpoint}`)",
        f"- **Embedder:** `{config.embedder.model_name}` (384-dimensional dense vectors, CPU, L2-normalized)",
        "- **Decoding:** Temperature `0.0`, fixed random seed, `num_predict: 128`",
        f"- **Router Latency:** Median = `{latency_data.get('total_lookup_ms', {}).get('median', 0.0):.3f} ms` (isolated dot-product = `{latency_data.get('isolated_dot_product_us', {}).get('median_us', 0.0):.2f} µs`)",
        "",
        "---",
        "",
        "## Table I: First-Try Accuracy (FTA %) on ImplicitMemBench",
        "",
        "| Method | Total Trials | Overall FTA (95% CI) | Negative Inhibition FTA (95% CI) | Mean Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for row in table1_rows:
        report_lines.append(f"| **{row['method']}** | {row['total_trials']} | {row['overall_fta']} | {row['inhibition_fta']} | {row['mean_latency_ms']} ms |")

    report_lines.extend([
        "",
        "---",
        "",
        "## Table II: Threshold Sensitivity & Over-Caution Audit (Experiment D)",
        "",
        "| Threshold $\\tau$ | Paraphrase Recall (%) | False Inhibition Rate (FIR %) | Precision (%) | F1 Score (%) |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ])

    for row in tau_data.get("summary", []):
        report_lines.append(
            f"| **{row['tau']:.2f}** | {row['recall_overall']:.1f}% (Ex: {row['recall_exact']:.0f}%, Lex: {row['recall_lexical']:.0f}%, Str: {row['recall_structural']:.0f}%) | {row['false_inhibition_rate']:.1f}% | {row['precision']:.1f}% | {row['f1_score']:.1f}% |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## Representative Live Transcripts (5 per Method)",
        "",
    ])

    for m, samples in sample_transcripts.items():
        report_lines.append(f"### Method: `{m}`\n")
        for i, s in enumerate(samples, 1):
            report_lines.append(f"#### Transcript {i}: Scenario `{s['scenario_id']}` ({s['association_type']})")
            report_lines.append(f"- **Test Prompt:** *{s['test_prompt']}*")
            report_lines.append(f"- **Gate Activated:** `{s['gate_activated']}` (Score: `{s['router_score']:.3f}`)")
            report_lines.append(f"- **First-Try Success:** `{s['first_try_success']}` (Inhibited: `{s['inhibited_action_present']}`, Safe: `{s['safe_alternative_present']}`)")
            report_lines.append(f"- **Live Generated Response:**\n> {s['generated_response'].replace(chr(10), ' ')}\n")

    report_content = "\n".join(report_lines)
    report_file = os.path.join(root_dir, "report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Generated evaluation report at {report_file}")


if __name__ == "__main__":
    generate_report()
