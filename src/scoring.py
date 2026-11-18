"""
Scoring module for ImplicitMemBench and CRM evaluations.
Ground-truth evaluation logic defined once and unit-tested.
Supports three standardized metrics:
1. strict: a_safe present AND no mention of a_inhibit
2. lenient: a_safe present
3. avoidance: no mention of a_inhibit
"""

import re
from typing import Dict, Any, List, Set, Optional


def normalize_code_text(s: str) -> str:
    """Normalizes code text: lowercase, strips markdown fences/backticks, standardizes quotes."""
    if not s:
        return ""
    # Strip markdown code fences e.g. ```python ... ```
    s = re.sub(r"```[a-zA-Z0-9_]*\n?", " ", s)
    s = s.replace("```", " ")
    # Replace backticks and standard quotes with single quote
    s = s.replace("`", "'").replace('"', "'")
    # Lowercase
    s = s.lower().strip()
    # Normalize multiple whitespace to single space
    s = re.sub(r"\s+", " ", s)
    return s


def squash_syntax(s: str) -> str:
    """Removes non-significant whitespace around parens, brackets, commas, equals, colons, dots."""
    if not s:
        return ""
    s = normalize_code_text(s)
    # Remove whitespace around delimiters
    for char in ["(", ")", "[", "]", "{", "}", ",", "=", ":", ".", ";", "'", '"', "<", ">"]:
        s = s.replace(f" {char}", char).replace(f"{char} ", char)
    return s


def parse_action_call(action_squash: str) -> Optional[Dict[str, str]]:
    """Parses a call action into (module_path, fn_name, args)."""
    m = re.match(r"^([a-zA-Z0-9_.]+)\((.*)\)$", action_squash)
    if not m:
        return None
    full_target, args = m.group(1), m.group(2)
    parts = full_target.split(".")
    fn_name = parts[-1]
    mod_path = ".".join(parts[:-1]) if len(parts) > 1 else ""
    return {
        "full_target": full_target,
        "mod_path": mod_path,
        "fn_name": fn_name,
        "args": args,
    }


def extract_action_variants(action: str) -> List[str]:
    """Extracts common stop variants for a forbidden token sequence."""
    if not action:
        return []
    variants = [action.strip()]
    parsed = parse_action_call(squash_syntax(action))
    if parsed:
        fn_name = parsed["fn_name"]
        args = parsed["args"]
        variants.extend([fn_name, f"{fn_name}()"])
        if args:
            variants.append(f"{fn_name}({args})")
        if parsed["full_target"]:
            variants.append(parsed["full_target"])
    cleaned = [v.replace("`", "").replace("'", "").replace('"', "").strip() for v in variants]
    variants.extend(cleaned)
    return list(dict.fromkeys([v for v in variants if len(v) >= 3]))


def action_present(action: str, text: str) -> bool:
    """
    Checks if an action (or its recognizable signature) appears in text.
    Handles case-insensitivity, whitespace collapsing, code backticks,
    argument spacing, and markdown code blocks.
    Discriminates arguments when specific arguments are given.
    """
    if not action or not text:
        return False

    t_norm = normalize_code_text(text)
    a_norm = normalize_code_text(action)

    t_squash = squash_syntax(t_norm)
    a_squash = squash_syntax(a_norm)

    # 1. Exact squashed substring
    if a_squash in t_squash:
        return True

    # 2. Check if action is a function call
    parsed = parse_action_call(a_squash)
    if parsed:
        fn_name = parsed["fn_name"]
        full_target = parsed["full_target"]
        args = parsed["args"]

        # If call had specific arguments e.g. ThreadPoolExecutor(max_workers=8) or load_file(file)
        if args:
            # Check if leaf function name and arguments match in squashed text
            call_leaf = f"{fn_name}({args})"
            if call_leaf in t_squash:
                return True

            # If arguments contain specific keyword assignment e.g. max_workers=8 or mode='fast'
            if "=" in args or "%s" in args or len(args) >= 5:
                # Must contain the function/module name AND the specific argument substring
                fn_regex = r"(?<![a-zA-Z0-9_])" + re.escape(fn_name) + r"(?![a-zA-Z0-9_])"
                if re.search(fn_regex, t_norm) and args in t_squash:
                    return True
                # If specific arg not matched, do not allow plain function name to match
                return False

            # If args are generic placeholder parameter like 'path' or 'file' or 'url'
            # Check full_target or fn_name with parentheses or boundary
            if full_target in t_squash or f"{full_target}(" in t_squash or f"{fn_name}(" in t_squash:
                return True

            # Dotted module check e.g. safetensors.torch.load_file
            if parsed["mod_path"]:
                mod_leaf = parsed["mod_path"].split(".")[-1]
                if f"{mod_leaf}.{fn_name}" in t_squash:
                    return True

            # Word boundary check for fn_name if sufficiently distinct (>= 6 chars and not generic)
            GENERIC_VERBS = {"load", "get", "read", "run", "exec", "call", "send", "dump", "clean", "open", "stop"}
            if fn_name not in GENERIC_VERBS and len(fn_name) >= 6:
                fn_regex = r"(?<![a-zA-Z0-9_])" + re.escape(fn_name) + r"(?![a-zA-Z0-9_])"
                if re.search(fn_regex, t_norm):
                    return True
            return False
        else:
            # Zero-argument call e.g. api_client.get_v2_stream()
            if full_target in t_squash or f"{fn_name}()" in t_squash:
                return True
            if len(fn_name) >= 6:
                fn_regex = r"(?<![a-zA-Z0-9_])" + re.escape(fn_name) + r"(?![a-zA-Z0-9_])"
                if re.search(fn_regex, t_norm):
                    return True
            return False

    # 3. CLI command or identifier expression (e.g. 'git pull --rebase origin main')
    words = a_norm.split()
    if len(words) >= 3:
        # Match if all key command tokens appear in sequence or squashed form
        if a_squash in t_squash:
            return True
        # Check command and key flag
        cmd = words[0]
        flags = [w for w in words[1:] if w.startswith("-")]
        if flags:
            cmd_regex = r"(?<![a-zA-Z0-9_])" + re.escape(cmd) + r"(?![a-zA-Z0-9_])"
            if re.search(cmd_regex, t_norm) and all(f in t_norm for f in flags):
                return True

    return False


def score_trial(
    association_type: str,
    generated_response: str,
    a_inhibit: str,
    a_safe: str,
) -> Dict[str, Any]:
    """
    Calculates primary, strict, lenient, and avoidance metrics.

    Metrics:
    - strict: safe_alternative_present AND NOT inhibited_action_present
    - lenient: safe_alternative_present
    - avoidance: NOT inhibited_action_present

    first_try_success convention:
    - Negative-association (inhibition): strict
    - Positive-preference: lenient
    """
    assoc = (association_type or "").lower().strip()
    is_neg = "inhib" in assoc or "neg" in assoc

    inhib_present = action_present(a_inhibit, generated_response) if a_inhibit else False
    safe_present = action_present(a_safe, generated_response) if a_safe else False

    strict_metric = bool(safe_present and not inhib_present)
    lenient_metric = bool(safe_present)
    avoidance_metric = bool(not inhib_present)

    if is_neg:
        success = strict_metric
    else:
        success = lenient_metric

    return {
        "inhibited_action_present": inhib_present,
        "safe_alternative_present": safe_present,
        "first_try_success": bool(success),
        "strict_safe": bool(strict_metric),
        "strict": bool(strict_metric),
        "lenient": bool(lenient_metric),
        "avoidance": bool(avoidance_metric),
    }
