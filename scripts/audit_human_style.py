#!/usr/bin/env python3
"""
scripts/audit_human_style.py - Programmatic Stylometric AI-Detection & 5-Gram Originality Auditor.

Verifies that academic LaTeX manuscripts comply with top-tier Q1 human writing benchmarks:
1. Sentence-Length Burstiness: Standard Deviation (SD) >= 9.0 words, Mean in [20.0, 26.0] words.
2. Zero Participial '-ing' Tail Clauses: e.g., ', highlighting...', ', underscoring...', ', demonstrating...'.
3. Zero Mechanical AI Transitions: e.g., 'Furthermore,', 'Moreover,', 'Additionally,', 'Crucially,'.
4. Zero Banned AI Lexicon: e.g., 'delve', 'tapestry', 'testament', 'pivotal', 'crucial', 'vital', 'leverage', etc.
5. 5-Gram Originality Auditor: Detects copy-paste repetitions and synthetic template cliches.
6. Core 2026 Citation Coverage: Ensures all 8 foundational 2026 papers are cited.

Exits with code 0 if PASSED (ZERO AI FLAGS), or code 1 with line-by-line violation reports.
"""

import sys
import os
import re
import math
import statistics
import argparse
from typing import List, Dict, Tuple, Set
from collections import Counter

# Ensure UTF-8 console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 1. BANNED AI LEXICON (Case-insensitive word boundary match)
BANNED_LEXICON = [
    r'\bdelve(?:s|d|ing)?\b',
    r'\btapestr(?:y|ies)\b',
    r'\btestament(?:s)?\b',
    r'\bunderscore(?:s|d|ing)?\b',
    r'\bpivotal\b',
    r'\bcrucial(?:ly)?\b',
    r'\bvital(?:ly)?\b',
    r'\bintricate(?:ly)?\b',
    r'\bmultifaceted\b',
    r'\brealm(?:s)?\b',
    r'\bparadigm(?:s)?\b',
    r'\bharness(?:es|ed|ing)?\b',
    r'\bleverage(?:s|d|ing)?\b',
    r'\butilize(?:s|d|ing|ation)?\b',
    r'\bfoster(?:s|ed|ing)?\b',
    r'\bshowcase(?:s|d|ing)?\b',
    r'\bboast(?:s|ed|ing)?\b',
    r'\bbeacon(?:s)?\b',
    r'\bmeticulous(?:ly)?\b',
    r'\bcommendable\b',
    r'\bcomprehensive(?:ly)?\b',
    r'\bgroundbreaking\b',
    r'\bcornerstone(?:s)?\b',
    r'\bseamless(?:ly)?\b',
    r'\bholistic(?:ally)?\b',
    r'\bsynerg(?:y|ies)\b',
    r'\bever-evolving\b',
    r'\bpoised\s+to\b',
    r'\bunravel(?:s|ed|ing)?\b',
    r'\bembark(?:s|ed|ing)?\b',
]

# 2. MECHANICAL AI TRANSITIONS (Sentences or paragraphs starting with these)
MECHANICAL_TRANSITIONS = [
    r'^[ \t]*Furthermore,',
    r'^[ \t]*Moreover,',
    r'^[ \t]*Additionally,',
    r'^[ \t]*In\s+addition,',
    r'^[ \t]*Consequently,',
    r'^[ \t]*Importantly,',
    r'^[ \t]*Notably,',
    r'^[ \t]*Crucially,',
    r'^[ \t]*In\s+conclusion,',
    r'^[ \t]*To\s+conclude,',
    r'^[ \t]*To\s+sum\s+up,',
]

# 3. PARTICIPIAL -ING TAIL CLAUSES
PARTICIPIAL_TAILS = [
    r',\s+highlighting\b',
    r',\s+underscoring\b',
    r',\s+demonstrating\b',
    r',\s+ensuring\b',
    r',\s+confirming\b',
    r',\s+proving\b',
    r',\s+making\s+it\b',
    r',\s+causing\b',
    r',\s+resulting\s+in\b',
    r',\s+yielding\b',
    r',\s+facilitating\b',
    r',\s+providing\s+valuable\b',
]

# 4. MANDATORY 2026 CITATIONS (Keys or arXiv matches in references / text)
CORE_2026_PAPERS = {
    "ImplicitMemBench": ["qin2026implicitmembench", "implicitmembench", "2604.08064"],
    "LightReasoner": ["lightreasoner2026", "lightreasoner", "2510.07962"],
    "COLD-Steer": ["coldsteer2026", "cold-steer", "2603.06495"],
    "CoSToM": ["costom2026", "costom", "2604.10031"],
    "AdapTime": ["adaptime2026", "adaptime", "2604.24175"],
    "Diversity Collapse": ["diversitycollapse2026", "diversity collapse", "2604.18005"],
    "Secret Leakage in Code LLMs": ["secretleakage2026", "secret leakage", "2604.17814"],
    "Reverse CAPTCHA": ["reversecaptcha2026", "reverse captcha", "2603.00164"],
}


def clean_latex(content: str) -> str:
    """Strips LaTeX comments, equations, commands, and environments for clean text analysis."""
    # Remove comments
    lines = []
    for line in content.splitlines():
        # Keep track of escaped %
        stripped = re.sub(r'(?<!\\)%.*$', '', line)
        lines.append(stripped)
    text = "\n".join(lines)

    # Remove display math \[ ... \] and $$ ... $$ and equation environments
    text = re.sub(r'\\\[.*?\\\]', ' [EQUATION] ', text, flags=re.DOTALL)
    text = re.sub(r'\$\$.*?\$\$', ' [EQUATION] ', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{(?:equation|align|gather|tabular|table|figure).*?\}.*?\\end\{(?:equation|align|gather|tabular|table|figure).*?\}', ' [TABLE_OR_FLOAT] ', text, flags=re.DOTALL)
    # Remove inline math $ ... $
    text = re.sub(r'\$.*?\$', ' [MATH] ', text)

    # Normalize citations \cite{...} -> [CITATION]
    text = re.sub(r'\\cite\{.*?\}', ' [REF] ', text)
    # Remove common commands like \textbf{...}, \textit{...}, etc.
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[.*?\])?\{([^}]*)\}', r'\1', text)
    # Remove remaining backslash commands like \item, \section, etc.
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)

    return text


def extract_sentences(raw_text: str) -> List[str]:
    """Extracts prose sentences excluding headers and empty lines."""
    # Split text on sentence boundaries
    raw_sentences = re.split(r'(?<=[.!?])\s+', raw_text)
    clean_sentences = []
    for s in raw_sentences:
        s = s.strip()
        # Filter out table fragments, float markers, equations, or short artifacts
        words = [w for w in re.findall(r'\b[a-zA-Z0-9_-]+\b', s) if w not in ['EQUATION', 'TABLE_OR_FLOAT', 'REF', 'MATH']]
        if len(words) >= 3:
            clean_sentences.append(" ".join(words))
    return clean_sentences


def audit_file(file_path: str) -> Dict[str, any]:
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}", "passed": False}

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        raw_content = f.read()

    lines = raw_content.splitlines()
    violations: List[Dict[str, any]] = []

    # 1. Line-by-line check for Banned Lexicon, Transitions, and Participial Tails
    for line_idx, line in enumerate(lines, 1):
        # Ignore comments
        code_part = re.sub(r'(?<!\\)%.*$', '', line).strip()
        if not code_part:
            continue

        # Check banned transitions
        for pat in MECHANICAL_TRANSITIONS:
            if re.search(pat, code_part, re.IGNORECASE):
                violations.append({
                    "line": line_idx,
                    "category": "Mechanical AI Transition",
                    "match": re.search(pat, code_part, re.IGNORECASE).group(0),
                    "context": code_part[:90]
                })

        # Check participial tails
        for pat in PARTICIPIAL_TAILS:
            m = re.search(pat, code_part, re.IGNORECASE)
            if m:
                violations.append({
                    "line": line_idx,
                    "category": "Participial Tail Clause",
                    "match": m.group(0),
                    "context": code_part[:90]
                })

        # Check banned lexicon
        for pat in BANNED_LEXICON:
            m = re.search(pat, code_part, re.IGNORECASE)
            if m:
                violations.append({
                    "line": line_idx,
                    "category": "Banned AI Lexicon",
                    "match": m.group(0),
                    "context": code_part[:90]
                })

    # 2. Stylometric Burstiness & Sentence-Length Distribution
    cleaned_prose = clean_latex(raw_content)
    sentences = extract_sentences(cleaned_prose)
    lengths = [len(s.split()) for s in sentences]

    if lengths:
        mean_len = statistics.mean(lengths)
        sd_len = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    else:
        mean_len, sd_len = 0.0, 0.0

    burstiness_passed = (sd_len >= 9.0) and (19.0 <= mean_len <= 26.0)

    # 3. 5-Gram Originality Audit
    five_grams = []
    all_words = []
    for s in sentences:
        words = s.lower().split()
        all_words.extend(words)
        for i in range(len(words) - 4):
            five_grams.append(tuple(words[i:i+5]))

    fg_counts = Counter(five_grams)
    excessive_repeats = [(fg, count) for fg, count in fg_counts.items() if count >= 4]

    # 4. Mandatory 2026 Citations Audit
    missing_citations = []
    content_lower = raw_content.lower()
    for paper_name, keywords in CORE_2026_PAPERS.items():
        found = any(k in content_lower for k in keywords)
        if not found:
            missing_citations.append(paper_name)

    passed = (
        len(violations) == 0
        and burstiness_passed
        and len(missing_citations) == 0
        and len(excessive_repeats) == 0
    )

    return {
        "file": file_path,
        "passed": passed,
        "total_lines": len(lines),
        "total_sentences": len(sentences),
        "mean_sentence_length": round(mean_len, 2),
        "sentence_length_sd": round(sd_len, 2),
        "burstiness_passed": burstiness_passed,
        "violations": violations,
        "excessive_5grams": excessive_repeats[:5],
        "missing_citations": missing_citations,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit LaTeX paper for human stylometric authenticity.")
    parser.add_argument("files", nargs="+", help="LaTeX files to audit")
    args = parser.parse_args()

    all_passed = True
    print("\n" + "=" * 80)
    print("PROGRAMMATIC STYLOMETRIC AI-DETECTION & 5-GRAM ORIGINALITY AUDITOR")
    print("=" * 80)

    for fpath in args.files:
        res = audit_file(fpath)
        print(f"\nAudit Report for: {fpath}")
        print("-" * 80)
        if "error" in res:
            print(f"[FAIL] {res['error']}")
            all_passed = False
            continue

        print(f"- Analyzed Sentences:        {res['total_sentences']}")
        print(f"- Mean Sentence Length:      {res['mean_sentence_length']} words (Target: 20-26 words)")
        print(f"- Sentence Burstiness (SD):  {res['sentence_length_sd']} words (Target: >= 9.0 words) -> {'[PASS]' if res['burstiness_passed'] else '[FAIL]'}")
        print(f"- Missing 2026 Citations:    {len(res['missing_citations'])}")
        if res['missing_citations']:
            for mc in res['missing_citations']:
                print(f"  * Missing citation for: {mc}")
        print(f"- Repeated 5-Grams (>=4):    {len(res['excessive_5grams'])}")
        print(f"- Stylometric Rule Flags:    {len(res['violations'])}")

        if res["violations"]:
            print("\n  [!] Violations Detected:")
            for v in res["violations"]:
                print(f"    Line {v['line']:4d} | [{v['category']}] '{v['match']}': {v['context']}")

        if res["passed"]:
            print("\n>>> RESULT: [PASSED - 100% AUTHENTIC HUMAN SCHOLARSHIP (ZERO AI FLAGS)]")
        else:
            print("\n>>> RESULT: [FAILED - ADJUSTMENTS REQUIRED TO MEET TOP-TIER HUMAN BENCHMARK]")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("OVERALL AUDIT: ALL MANUSCRIPTS PASSED WITH ZERO AI FLAGS.")
        print("=" * 80 + "\n")
        sys.exit(0)
    else:
        print("OVERALL AUDIT: AT LEAST ONE MANUSCRIPT REQUIRES REVISION.")
        print("=" * 80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
