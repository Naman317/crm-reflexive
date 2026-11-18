"""
Unit tests for data integrity.
Enforces:
1. Paraphrase inequality: exact != lexical != structural for all scenarios.
2. Classical dataset balance: 50 inhibition + 50 preference, tool/system/conversational domain counts.
3. Procedural dataset size: 100 items.
4. Benign controls: 100 items.
5. Distractor pool: at least 150 items.
"""

import os
import json
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def test_classical_dataset_integrity():
    p = os.path.join(DATA_DIR, "implicit_mem_bench_classical_100.json")
    assert os.path.exists(p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 100
    inhibit_count = sum(1 for d in data if "inhib" in d["association_type"].lower())
    pref_count = sum(1 for d in data if "pref" in d["association_type"].lower())
    assert inhibit_count == 50
    assert pref_count == 50

    domains = [d["domain"] for d in data]
    assert domains.count("tool_safety") == 34
    assert domains.count("system_protection") == 33
    assert domains.count("conversational_adaptation") == 33


def test_procedural_dataset_integrity():
    p = os.path.join(DATA_DIR, "implicit_mem_bench_procedural_100.json")
    assert os.path.exists(p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 100


def test_paraphrase_inequality():
    p = os.path.join(DATA_DIR, "paraphrase_stress_test_300.json")
    assert os.path.exists(p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 300
    by_scen = {}
    for probe in data:
        scen = probe["scenario_id"]
        ptype = probe["paraphrase_type"]
        text = probe["test_prompt"].strip().lower()
        by_scen.setdefault(scen, {})[ptype] = text

    assert len(by_scen) == 100
    for scen, tiers in by_scen.items():
        assert "exact" in tiers
        assert "lexical" in tiers
        assert "structural" in tiers
        assert tiers["exact"] != tiers["lexical"], f"Exact == Lexical in {scen}"
        assert tiers["exact"] != tiers["structural"], f"Exact == Structural in {scen}"
        assert tiers["lexical"] != tiers["structural"], f"Lexical == Structural in {scen}"


def test_benign_controls_integrity():
    p = os.path.join(DATA_DIR, "benign_overcaution_controls_100.json")
    assert os.path.exists(p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 100
    for item in data:
        assert item["should_inhibit"] is False


def test_distractor_pool_integrity():
    p = os.path.join(DATA_DIR, "distractor_pool_150.json")
    assert os.path.exists(p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) >= 150
    for item in data:
        assert "question" in item
        assert "answer" in item
        assert len(item["question"]) > 5
        assert len(item["answer"]) > 5
