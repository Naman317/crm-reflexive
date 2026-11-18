import json
from scipy.stats import binomtest

with open("legacy/prior_run_20261115/main_benchmark_results.json", "r", encoding="utf-8") as f:
    recs = json.load(f)

by_key = {}
for r in recs:
    key = (r["scenario_id"], r["seed"])
    by_key.setdefault(key, {})[r["method"]] = r

def mcnemar(m1, m2, metric="first_try_success"):
    b = 0
    c = 0
    for key, methods in by_key.items():
        if m1 in methods and m2 in methods:
            s1 = bool(methods[m1][metric])
            s2 = bool(methods[m2][metric])
            if s1 and not s2:
                b += 1
            elif not s1 and s2:
                c += 1
    disc = b + c
    p_val = binomtest(b, disc, 0.5).pvalue if disc > 0 else 1.0
    return b, c, disc, p_val

for metric in ["first_try_success", "strict_safe"]:
    print(f"=== Metric: {metric} ===")
    for m1, m2 in [("crm_prompt", "vanilla"), ("crm_prompt", "mem0_rag"), ("crm_mask", "vanilla"), ("crm_mask", "crm_prompt")]:
        b, c, disc, p = mcnemar(m1, m2, metric)
        print(f"{m1} vs {m2}: b={b}, c={c}, disc={disc}, p={p:.6e}")
