import os
import sys
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from checks import run_full_anti_stub_audit

# Load available test records
m120_path = os.path.join(root_dir, "results", "raw_logs", "m120_sample_10.json")
records = []
if os.path.exists(m120_path):
    with open(m120_path, "r", encoding="utf-8") as f:
        records.extend(json.load(f))

dev_path = os.path.join(root_dir, "results", "raw_logs", "dev_selection.jsonl")
if os.path.exists(dev_path):
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

print(f"Total merged audit records: {len(records)}")
passed = run_full_anti_stub_audit(
    records=records,
    src_root="src",
    check_horizon=False,
    require_balanced=False,
    results_dir="results",
    legacy_dir="legacy",
)
print("AUDIT RETURN CODE:", passed)
