"""
Script to create strict DEV (40) and TEST (160) task splits.
DEV: 20 Classical (10 inhibition + 10 preference) + 20 Procedural = 40 tasks.
TEST: 80 Classical (40 inhibition + 40 preference) + 80 Procedural = 160 tasks.
"""

import os
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(root_dir, "data")

with open(os.path.join(data_dir, "implicit_mem_bench_classical_100.json"), "r", encoding="utf-8") as f:
    classical = json.load(f)

with open(os.path.join(data_dir, "implicit_mem_bench_procedural_100.json"), "r", encoding="utf-8") as f:
    procedural = json.load(f)

# Split Classical: 20 DEV (10 inhib, 10 pref) and 80 TEST (40 inhib, 40 pref)
classical_inhib = [t for t in classical if "inhib" in t.get("association_type", "").lower()]
classical_pref = [t for t in classical if "pref" in t.get("association_type", "").lower()]

dev_classical = classical_inhib[:10] + classical_pref[:10]
test_classical = classical_inhib[10:] + classical_pref[10:]

# Split Procedural: 20 DEV and 80 TEST
dev_procedural = procedural[:20]
test_procedural = procedural[20:]

dev_tasks = dev_classical + dev_procedural
test_tasks = test_classical + test_procedural

print(f"DEV Tasks count: {len(dev_tasks)} (Classical: {len(dev_classical)}, Procedural: {len(dev_procedural)})")
print(f"TEST Tasks count: {len(test_tasks)} (Classical: {len(test_classical)}, Procedural: {len(test_procedural)})")

assert len(dev_tasks) == 40
assert len(test_tasks) == 160

# Save datasets
dev_path = os.path.join(data_dir, "dev_tasks_40.json")
test_path = os.path.join(data_dir, "test_tasks_160.json")

with open(dev_path, "w", encoding="utf-8") as f:
    json.dump(dev_tasks, f, indent=2)

with open(test_path, "w", encoding="utf-8") as f:
    json.dump(test_tasks, f, indent=2)

print(f"Saved DEV tasks to {dev_path}")
print(f"Saved TEST tasks to {test_path}")
