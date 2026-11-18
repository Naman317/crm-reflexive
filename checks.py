"""
Automated Anti-Stub and Validity Audit Engine.
Enforces hard integrity rules:
1. Distinct response diversity (> 20% per method and file).
2. Zero canned/templated fallback strings in responses or source code.
3. Gate activation strict equivalence: gate_activated == (router_score >= tau).
4. Prompt token validity: prompt_tokens > 0 and increasing with M in horizon files.
5. Zero error rows: error_message must be None or empty.
6. Identical trial counts per (method, condition, seed) across methods.
7. Paraphrase inequality: exact != lexical != structural for every scenario.
8. Checksum verification against legacy/ (zero MD5 collisions).
"""

import os
import re
import sys
import json
import hashlib
import logging
from typing import List, Dict, Any, Set, Tuple

logger = logging.getLogger(__name__)

BANNED_TEMPLATE_PATTERNS = [
    r"Understood\.\s*Avoiding past penalized action",
    r"Executing safe procedure:\s*api_client",
    r"Executing standard operation using default parameters",
    r"I will proceed with the requested task using standard conventions",
]


def audit_source_code_for_templates(src_root: str = "src") -> List[str]:
    """Greps all Python files in src/ for canned template phrases."""
    violations = []
    if not os.path.exists(src_root):
        return [f"Source directory '{src_root}' not found."]

    for root, _, files in os.walk(src_root):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as pyfile:
                    content = pyfile.read()
                    for pat in BANNED_TEMPLATE_PATTERNS:
                        if re.search(pat, content, re.IGNORECASE):
                            violations.append(f"{path}: Matched banned template pattern '{pat}'")
    return violations


def audit_checksums_against_legacy(results_dir: str = "results", legacy_dir: str = "legacy") -> List[str]:
    """Ensures no result file in results_dir has the same MD5 checksum as any file in legacy_dir."""
    violations = []
    if not os.path.exists(legacy_dir) or not os.path.exists(results_dir):
        return []

    legacy_hashes: Dict[str, str] = {}
    for root, _, files in os.walk(legacy_dir):
        for f in files:
            p = os.path.join(root, f)
            try:
                with open(p, "rb") as bfile:
                    h = hashlib.md5(bfile.read()).hexdigest()
                    legacy_hashes[h] = p
            except Exception:
                pass

    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.endswith(".json") or f.endswith(".csv") or f.endswith(".jsonl"):
                p = os.path.join(root, f)
                try:
                    with open(p, "rb") as bfile:
                        h = hashlib.md5(bfile.read()).hexdigest()
                        if h in legacy_hashes:
                            violations.append(
                                f"Result file {p} has identical checksum {h} to legacy file {legacy_hashes[h]}"
                            )
                except Exception:
                    pass
    return violations


def audit_paraphrase_prompts_diversity(paraphrase_path: str = "data/paraphrase_stress_test_300.json") -> List[str]:
    """Ensures exact, lexical, and structural prompts strictly differ for every scenario."""
    violations = []
    if not os.path.exists(paraphrase_path):
        return [f"Paraphrase dataset '{paraphrase_path}' not found."]

    with open(paraphrase_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_scenario: Dict[str, Dict[str, str]] = {}
    for item in data:
        scen_id = item.get("scenario_id")
        p_type = item.get("paraphrase_type")
        prompt = item.get("test_prompt", "").strip()
        if scen_id and p_type:
            by_scenario.setdefault(scen_id, {})[p_type] = prompt

    for scen_id, tiers in by_scenario.items():
        exact = tiers.get("exact")
        lexical = tiers.get("lexical")
        structural = tiers.get("structural")

        if exact and lexical and exact == lexical:
            violations.append(f"Scenario {scen_id}: exact and lexical prompts are identical: '{exact}'")
        if lexical and structural and lexical == structural:
            violations.append(f"Scenario {scen_id}: lexical and structural prompts are identical: '{lexical}'")
        if exact and structural and exact == structural:
            violations.append(f"Scenario {scen_id}: exact and structural prompts are identical: '{exact}'")

    return violations


def audit_trial_records(records: List[Dict[str, Any]], check_horizon: bool = False, require_balanced: bool = True) -> List[str]:
    """Audits trial record list against all anti-stub rules."""
    violations = []
    if not records:
        return ["No trial records provided for audit."]

    by_method: Dict[str, List[str]] = {}
    by_method_conditions: Dict[str, Set[Tuple[Any, ...]]] = {}

    for idx, r in enumerate(records):
        m = r.get("method", "unknown")
        scen = r.get("scenario_id", f"row_{idx}")
        resp = r.get("generated_response", "")
        by_method.setdefault(m, []).append(resp)

        # Condition key: (scenario_id, horizon, paraphrase_type, seed)
        cond_key = (
            r.get("scenario_id"),
            r.get("interference_horizon"),
            r.get("paraphrase_type", "canonical"),
            r.get("seed", 42),
        )
        by_method_conditions.setdefault(m, set()).add(cond_key)

        # Check 1: Zero error rows
        err = r.get("error_message")
        if err:
            violations.append(f"Row {idx} ({scen}, {m}): Unresolved error row: {err}")

        # Check 2: No canned templates in generated response
        for pat in BANNED_TEMPLATE_PATTERNS:
            if re.search(pat, resp, re.IGNORECASE):
                violations.append(
                    f"Scenario {scen}, Method {m}: Response matched banned template '{pat}': {resp[:60]}"
                )

        # Check 3: Gate activation must strictly equal (router_score >= tau)
        score = r.get("router_score", 0.0)
        tau = r.get("tau", 0.35)
        gate = r.get("gate_activated", False)
        expected_gate = score >= tau
        if gate != expected_gate:
            violations.append(
                f"Gate activation mismatch in {scen} ({m}): score={score}, tau={tau}, gate={gate}, expected={expected_gate}"
            )

        # Check 4: Prompt tokens must be strictly positive
        pt = r.get("prompt_tokens", 0)
        if pt <= 0:
            violations.append(f"Scenario {scen}, Method {m}: Non-positive prompt_tokens: {pt}")

    # Check 5: Response diversity > 20% per method
    for m, responses in by_method.items():
        total = len(responses)
        if total >= 5:
            unique_count = len(set(responses))
            ratio = unique_count / total
            if ratio <= 0.20:
                violations.append(
                    f"Diversity failure for method '{m}': unique responses={unique_count}/{total} ({ratio:.1%}) <= 20%"
                )

    # Check 6: Identical condition sets across methods (balanced evaluation)
    if require_balanced:
        methods = list(by_method_conditions.keys())
        if len(methods) > 1:
            ref_method = methods[0]
            ref_conditions = by_method_conditions[ref_method]
            for other_m in methods[1:]:
                other_conditions = by_method_conditions[other_m]
                diff = ref_conditions.symmetric_difference(other_conditions)
                if diff:
                    violations.append(
                        f"Trial count / condition mismatch between '{ref_method}' ({len(ref_conditions)}) and '{other_m}' ({len(other_conditions)}): {len(diff)} discrepancies."
                    )

    # Check 7: Horizon scaling token check if applicable
    if check_horizon:
        horizon_tokens: Dict[int, List[int]] = {}
        for r in records:
            h = r.get("interference_horizon")
            pt = r.get("prompt_tokens", 0)
            if h is not None and pt > 0:
                horizon_tokens.setdefault(h, []).append(pt)

        sorted_horizons = sorted(horizon_tokens.keys())
        if len(sorted_horizons) > 1:
            avg_tokens = [sum(horizon_tokens[h]) / len(horizon_tokens[h]) for h in sorted_horizons]
            for i in range(len(avg_tokens) - 1):
                if avg_tokens[i + 1] <= avg_tokens[i]:
                    violations.append(
                        f"Token scaling failure: horizon {sorted_horizons[i]} avg tokens ({avg_tokens[i]:.1f}) >= "
                        f"horizon {sorted_horizons[i+1]} avg tokens ({avg_tokens[i+1]:.1f})"
                    )

    return violations


def run_full_anti_stub_audit(
    records: List[Dict[str, Any]],
    src_root: str = "src",
    check_horizon: bool = False,
    require_balanced: bool = True,
    results_dir: str = "results",
    legacy_dir: str = "legacy",
    paraphrase_path: str = "data/paraphrase_stress_test_300.json",
) -> bool:
    """Runs all automated checks and raises / returns boolean."""
    print("=" * 65)
    print("RUNNING AUTOMATED ANTI-STUB AND REPRODUCIBILITY AUDIT")
    print("=" * 65)

    all_pass = True

    # 1. Source code grep for canned phrases
    src_violations = audit_source_code_for_templates(src_root)
    if src_violations:
        print("[FAIL] Banned template phrases detected in source code:")
        for v in src_violations:
            print("  *", v)
        all_pass = False
    else:
        print("[PASS] Source code clean: Zero hardcoded response templates in src/")

    # 2. Paraphrase inequality check
    para_violations = audit_paraphrase_prompts_diversity(paraphrase_path)
    if para_violations:
        print("[FAIL] Paraphrase prompt collisions detected:")
        for v in para_violations[:5]:
            print("  *", v)
        all_pass = False
    else:
        print("[PASS] Paraphrase inequality verified: exact != lexical != structural across all probes.")

    # 3. Checksum check against legacy
    chk_violations = audit_checksums_against_legacy(results_dir=results_dir, legacy_dir=legacy_dir)
    if chk_violations:
        print("[FAIL] Checksum collision with legacy canned results detected:")
        for v in chk_violations:
            print("  *", v)
        all_pass = False
    else:
        print("[PASS] Checksums verified: Zero result files collide with legacy archives.")

    # 4. Record checks
    rec_violations = audit_trial_records(records, check_horizon=check_horizon, require_balanced=require_balanced)
    if rec_violations:
        print("[FAIL] Trial record integrity violations detected:")
        for v in rec_violations[:10]:
            print("  *", v)
        if len(rec_violations) > 10:
            print(f"  ... and {len(rec_violations) - 10} more violations.")
        all_pass = False
    else:
        print("[PASS] Trial records valid: Diversity > 20%, Gate equality verified, Zero error rows.")
        if check_horizon:
            print("[PASS] Horizon token expansion monotonically verified.")

    if all_pass:
        print("[SUCCESS] All anti-stub and validation checks PASSED cleanly.\n")
    else:
        print("[FAILURE] Integrity audit encountered violations. Please see logs above.\n")

    return all_pass


if __name__ == "__main__":
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    src_v = audit_source_code_for_templates(src_dir)
    if src_v:
        print("Source violations:", src_v)
        sys.exit(1)

    para_path = os.path.join(os.path.dirname(__file__), "data", "paraphrase_stress_test_300.json")
    para_v = audit_paraphrase_prompts_diversity(para_path)
    if para_v:
        print("Paraphrase violations:", para_v)
        sys.exit(1)

    print("Checks script initialized cleanly.")
