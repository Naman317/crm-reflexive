"""
Data schemas for Contrastive Reflexive Memory (CRM) and ImplicitMemBench evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class ReflexRule:
    """
    Represents an extracted inhibitory reflex rule:
    r_i = (e_{CS, i}, P_{CS, i}, a_{inhibit, i}, V_{inhibit, i}, a_{safe, i})
    """
    rule_id: str
    trigger_pattern: str                       # P_{CS, i}: Conditioning stimulus cue/description
    inhibited_action: str                      # a_{inhibit, i}: Forbidden tool, parameter, path, or style
    inhibited_tokens: List[str]                # V_{inhibit, i}: Token strings to suppress at decoding
    safe_alternative: str                      # a_{safe, i}: Prescribed fallback action or parameter
    domain: str                                # E.g., 'tool_safety', 'system_protection', 'conversational'
    failure_cue: str                           # u_{US}: Error message, traceback, or user frustration marker
    embedding: Optional[np.ndarray] = None     # e_{CS, i}: L2-normalized dense embedding vector
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "trigger_pattern": self.trigger_pattern,
            "inhibited_action": self.inhibited_action,
            "inhibited_tokens": self.inhibited_tokens,
            "safe_alternative": self.safe_alternative,
            "domain": self.domain,
            "failure_cue": self.failure_cue,
            "metadata": self.metadata,
            "has_embedding": self.embedding is not None,
        }


@dataclass
class ConditioningScenario:
    """
    ImplicitMemBench scenario structure covering learning phase,
    distraction interference, and multi-tier test probes.
    """
    scenario_id: str
    scenario_type: str                         # 'classical' or 'procedural'
    association_type: str                      # 'inhibition' (negative) or 'preference' (positive)
    domain: str                                # Domain category
    learning_turns: List[Dict[str, str]]       # Conversation turns containing conditioned stimulus + US
    interference_turns: List[Dict[str, str]]   # Unrelated distractor turns of length M
    test_probe: str                            # Canonical test prompt
    paraphrases: Dict[str, str]                # {'exact': ..., 'lexical': ..., 'structural': ...}
    inhibited_action: str                      # Forbidden action string
    safe_alternative: str                      # Required safe substitute
    ground_truth_target: str                   # Expected gold target action
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type,
            "association_type": self.association_type,
            "domain": self.domain,
            "learning_turns": self.learning_turns,
            "interference_turns": self.interference_turns,
            "test_probe": self.test_probe,
            "paraphrases": self.paraphrases,
            "inhibited_action": self.inhibited_action,
            "safe_alternative": self.safe_alternative,
            "ground_truth_target": self.ground_truth_target,
            "metadata": self.metadata,
        }


@dataclass
class EvalResult:
    """
    Individual benchmark execution result evaluated under First-Try Accuracy (FTA).
    """
    scenario_id: str
    model_name: str
    method: str                                # 'vanilla', 'mem0_rag', 'crm_prompt', 'crm_mask'
    scenario_type: str
    association_type: str
    domain: str
    paraphrase_type: str                       # 'exact', 'lexical', 'structural', 'canonical'
    test_prompt: str
    generated_response: str
    router_score: float                        # Max cosine similarity s_t
    gate_activated: bool                       # s_t >= tau
    inhibited_action_present: bool             # Whether forbidden token/phrase leaked
    safe_alternative_present: bool             # Whether safe alternative was executed
    first_try_success: bool                    # not inhibited_action_present and safe_alternative_present
    interference_horizon: int                  # Distraction turns M
    latency_ms: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "model_name": self.model_name,
            "method": self.method,
            "scenario_type": self.scenario_type,
            "association_type": self.association_type,
            "domain": self.domain,
            "paraphrase_type": self.paraphrase_type,
            "test_prompt": self.test_prompt,
            "generated_response": self.generated_response,
            "router_score": round(self.router_score, 4),
            "gate_activated": self.gate_activated,
            "inhibited_action_present": self.inhibited_action_present,
            "safe_alternative_present": self.safe_alternative_present,
            "first_try_success": self.first_try_success,
            "interference_horizon": self.interference_horizon,
            "latency_ms": round(self.latency_ms, 2),
            "error_message": self.error_message,
        }
