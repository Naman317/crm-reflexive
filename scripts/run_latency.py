"""
Experiment E: Router Lookup Latency Measurement.
Measures router lookup time (embedding + matrix-vector dot product)
separately from model generation time over >= 1000 lookups.
Reports median, p90, p95, p99 in milliseconds.
"""

import os
import sys
import time
import json
import numpy as np

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import load_config
from src.embedder import SemanticTriggerEmbedder
from src.router import SemanticRouter


def run_latency_benchmark(num_lookups: int = 1000):
    config = load_config()
    print("=" * 70)
    print(f"RUNNING EXPERIMENT E: ROUTER LOOKUP LATENCY ({num_lookups} TRIALS)")
    print("=" * 70)

    embedder = SemanticTriggerEmbedder(model_name=config.embedder.model_name, device=config.embedder.device)
    router = SemanticRouter(embedder=embedder)

    # Load 100 classical rules
    classical_path = os.path.join(root_dir, config.paths.data_dir, "implicit_mem_bench_classical_100.json")
    with open(classical_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    router.clear()
    for t in tasks:
        trigger = t.get("learning_prompt", t.get("test_probe", ""))
        router.add_rule(
            scenario_id=t.get("scenario_id", ""),
            trigger_prompt=trigger,
            a_inhibit=t.get("inhibited_action", ""),
            a_safe=t.get("safe_alternative", ""),
            domain=t.get("domain", ""),
        )

    # Sample queries
    test_queries = [t.get("test_probe", "") for t in tasks]

    # Warmup
    for q in test_queries[:10]:
        router.route(q)

    # 1. Total Lookup Latency (Embedding + Dot Product)
    total_latencies_ms = []
    # 2. Isolated Matrix-Vector Dot Product Latency
    dot_latencies_ms = []

    # Pre-embed queries to measure isolated dot product
    pre_embedded = [embedder.embed(q) for q in test_queries]
    mat = router._ensure_matrix()

    for i in range(num_lookups):
        q = test_queries[i % len(test_queries)]
        q_emb = pre_embedded[i % len(pre_embedded)]

        # Time isolated dot product
        t_dot0 = time.perf_counter()
        scores = np.dot(mat, q_emb)
        best_idx = np.argmax(scores)
        t_dot1 = time.perf_counter()
        dot_latencies_ms.append((t_dot1 - t_dot0) * 1000.0)

        # Time full router lookup (embedding + matrix dot product)
        t0 = time.perf_counter()
        router.route(q)
        t1 = time.perf_counter()
        total_latencies_ms.append((t1 - t0) * 1000.0)

    total_arr = np.array(total_latencies_ms)
    dot_arr = np.array(dot_latencies_ms)

    metrics = {
        "num_lookups": num_lookups,
        "num_stored_rules": len(router.rules),
        "total_lookup_ms": {
            "mean": float(np.mean(total_arr)),
            "std": float(np.std(total_arr)),
            "median": float(np.median(total_arr)),
            "p90": float(np.percentile(total_arr, 90)),
            "p95": float(np.percentile(total_arr, 95)),
            "p99": float(np.percentile(total_arr, 99)),
        },
        "isolated_dot_product_us": {
            "mean_us": float(np.mean(dot_arr) * 1000.0),
            "median_us": float(np.median(dot_arr) * 1000.0),
            "p95_us": float(np.percentile(dot_arr, 95) * 1000.0),
        },
    }

    print("\n--- LATENCY BENCHMARK RESULTS ---")
    print(f"Stored Rules Matrix: {len(router.rules)} x 384")
    print(f"Full Lookup (Embed + Matrix Product):")
    print(f"  Median: {metrics['total_lookup_ms']['median']:.3f} ms")
    print(f"  P90:    {metrics['total_lookup_ms']['p90']:.3f} ms")
    print(f"  P95:    {metrics['total_lookup_ms']['p95']:.3f} ms")
    print(f"  P99:    {metrics['total_lookup_ms']['p99']:.3f} ms")
    print(f"Isolated Matrix-Vector Product:")
    print(f"  Median: {metrics['isolated_dot_product_us']['median_us']:.2f} microseconds")
    print(f"  P95:    {metrics['isolated_dot_product_us']['p95_us']:.2f} microseconds")

    out_dir = os.path.join(root_dir, config.paths.raw_logs_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "latency_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved latency results to {out_file}")
    return metrics


if __name__ == "__main__":
    run_latency_benchmark(num_lookups=1000)
