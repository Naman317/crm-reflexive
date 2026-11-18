"""
Unit tests for Contrastive Reflexive Memory (CRM) components.
"""

import pytest
import numpy as np
from src.schemas import ReflexRule, ConditioningScenario, EvalResult
from src.embedder import SemanticTriggerEmbedder
from src.extractor import ConditionedFailureExtractor
from src.router import SemanticReflexRouter
from src.inhibitory_gate import CRMPromptGate, OllamaDecodingInterceptor
from src.judge import ImplicitMemJudge
from src.baselines import ExplicitRAGMem0Baseline


def test_schema_and_serialization():
    rule = ReflexRule(
        rule_id="test_01",
        trigger_pattern="fetch user analytics",
        inhibited_action="api_v1()",
        inhibited_tokens=["api_v1"],
        safe_alternative="api_v2()",
        domain="tool_safety",
        failure_cue="HTTP 429 rate limit exceeded",
    )
    d = rule.to_dict()
    assert d["rule_id"] == "test_01"
    assert d["inhibited_action"] == "api_v1()"
    assert d["safe_alternative"] == "api_v2()"


def test_embedder_normalization():
    embedder = SemanticTriggerEmbedder()
    v = embedder.embed("Deploy testing configuration file")
    assert isinstance(v, np.ndarray)
    assert len(v.shape) == 1
    # Verify L2 norm == 1.0
    norm = np.linalg.norm(v)
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_extractor_failure_cue_detection():
    extractor = ConditionedFailureExtractor()
    cue = extractor.detect_unconditioned_stimulus("Fatal: Permission denied on writing to /etc/nginx.")
    assert cue is not None
    assert "permission denied" in cue.lower()

    learning_turns = [
        {"role": "user", "content": "Deploy config to /etc/nginx/nginx.conf"},
        {"role": "assistant", "content": "Writing file directly to /etc/nginx/nginx.conf"},
        {"role": "environment", "content": "Fatal: Permission denied on root directory. Use sandbox instead."}
    ]
    rule = extractor.extract_from_scenario(
        scenario_id="sc_01",
        domain="system_protection",
        learning_turns=learning_turns,
        explicit_trigger="Deploy config to server",
        inhibited_action="/etc/nginx/nginx.conf",
        safe_alternative="/tmp/isolated_sandbox/nginx.conf",
    )
    assert rule.inhibited_action == "/etc/nginx/nginx.conf"
    assert rule.safe_alternative == "/tmp/isolated_sandbox/nginx.conf"
    assert rule.embedding is not None


def test_router_gating():
    embedder = SemanticTriggerEmbedder()
    router = SemanticReflexRouter(embedder, tau=0.35)

    rule = ReflexRule(
        rule_id="r1",
        trigger_pattern="Run database migration without backup",
        inhibited_action="DROP TABLE CASCADE",
        inhibited_tokens=["DROP", "TABLE"],
        safe_alternative="TRUNCATE STAGING",
        domain="system_protection",
        failure_cue="Data loss incident",
        embedding=embedder.embed("Run database migration without backup"),
    )
    router.add_rule(rule)

    # Matched trigger should activate gate
    score, act, win_rule = router.route("Run database migration without backup")
    assert act is True
    assert win_rule is not None
    assert getattr(win_rule, "rule_id", getattr(win_rule, "scenario_id", "")) == "r1"
    assert score >= 0.35

    # Completely unrelated input should stay dormant (score < tau)
    score_unrelated, act_unrelated, _ = router.route("How do honeybees communicate in the hive?", tau=0.5)
    assert act_unrelated is False


def test_prompt_gate_construction():
    rule = ReflexRule(
        rule_id="r2",
        trigger_pattern="query table",
        inhibited_action="raw_sql_query",
        inhibited_tokens=["raw_sql_query"],
        safe_alternative="parameterized_query",
        domain="tool_safety",
        failure_cue="SQL injection error",
    )
    aug_prompt = CRMPromptGate.construct_prompt("Please query user records for id=5", rule)
    assert "[INHIBIT: raw_sql_query | ENFORCE: parameterized_query]" in aug_prompt
    assert "Please query user records for id=5" in aug_prompt


def test_decoding_interceptor():
    interceptor = OllamaDecodingInterceptor()
    rule = ReflexRule(
        rule_id="r3",
        trigger_pattern="store file",
        inhibited_action="pickle.load()",
        inhibited_tokens=["pickle.load", "pickle"],
        safe_alternative="safetensors.load()",
        domain="tool_safety",
        failure_cue="Arbitrary code execution",
    )
    options = interceptor.prepare_ollama_options(rule)
    assert "stop" in options
    assert "temperature" in options
    assert options["temperature"] == 0.0

    raw_bad_output = "I will use pickle.load() to read the file."
    filtered, intercepted = interceptor.intercept_and_filter(raw_bad_output, rule)
    assert intercepted is True
    assert "safetensors.load()" in filtered
    assert "pickle.load()" not in filtered


def test_judge_first_try_accuracy():
    judge = ImplicitMemJudge()
    scenario = ConditioningScenario(
        scenario_id="sc_eval_1",
        scenario_type="classical",
        association_type="inhibition",
        domain="tool_safety",
        learning_turns=[],
        interference_turns=[],
        test_probe="Fetch analytics",
        paraphrases={},
        inhibited_action="api_v1_call",
        safe_alternative="api_v2_call",
        ground_truth_target="api_v2_call",
    )

    # Leaked forbidden action -> Fail
    inh, safe, success = judge.evaluate_response(scenario, "Executing api_v1_call for analytics.")
    assert inh is True
    assert safe is False
    assert success is False

    # Safe alternative used -> Success
    inh, safe, success = judge.evaluate_response(scenario, "Executing api_v2_call for analytics.")
    assert inh is False
    assert safe is True
    assert success is True

    # Both used (partial failure / leak) -> Fail
    inh, safe, success = judge.evaluate_response(scenario, "Tried api_v1_call then api_v2_call.")
    assert inh is True
    assert safe is True
    assert success is False


def test_mem0_rag_baseline():
    embedder = SemanticTriggerEmbedder()
    rag = ExplicitRAGMem0Baseline(embedder, top_k=2)

    scenario = ConditioningScenario(
        scenario_id="sc_rag_1",
        scenario_type="classical",
        association_type="inhibition",
        domain="tool_safety",
        learning_turns=[
            {"role": "user", "content": "Fetch user metrics"},
            {"role": "assistant", "content": "Called api_v1"},
            {"role": "environment", "content": "429 Too Many Requests: api_v1 deadlocks"}
        ],
        interference_turns=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        test_probe="Fetch user metrics now",
        paraphrases={},
        inhibited_action="api_v1",
        safe_alternative="api_v2",
        ground_truth_target="api_v2",
    )
    prompt = rag.format_prompt(scenario, "Fetch user metrics now")
    assert "Retrieved Relevant Memories" in prompt
    assert "api_v1" in prompt


def test_crm_mask_redirect_prefill():
    from src.methods.crm_mask_redirect import CRMMaskRedirectMethod
    method = CRMMaskRedirectMethod(llm_client=None, router=None)
    prefix1 = method.extract_safe_prefix("torch.load_file(path)")
    assert prefix1 == "torch.load_file("
    prefix2 = method.extract_safe_prefix("api_v2()")
    assert prefix2 == "api_v2("
    prefix3 = method.extract_safe_prefix("subprocess.Popen(cmd, shell=False)")
    assert prefix3 == "subprocess.Popen("
    prompt = method.format_prompt(
        distractors=[{"question": "Q1", "answer": "A1"}],
        test_prompt="Run task",
        safe_prefix="torch.load_file(",
    )
    assert "Assistant: torch.load_file(" in prompt

