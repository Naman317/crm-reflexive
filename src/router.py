"""
Paraphrase-Invariant Semantic Router.
Evaluates s_t = max(e_t^T e_CS) >= tau against stored reflex buffer.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from .embedder import SemanticTriggerEmbedder


class ReflexRule:
    """Represents a conditioned inhibitory reflex tuple."""

    def __init__(
        self,
        scenario_id: str,
        trigger_prompt: str,
        embedding: np.ndarray,
        a_inhibit: str,
        a_safe: str,
        v_inhibit: Optional[List[int]] = None,
        domain: str = "",
    ):
        self.scenario_id = scenario_id
        self.trigger_prompt = trigger_prompt
        self.embedding = embedding  # Unit L2-norm (384,)
        self.a_inhibit = a_inhibit
        self.a_safe = a_safe
        self.v_inhibit = v_inhibit or []
        self.domain = domain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "trigger_prompt": self.trigger_prompt,
            "a_inhibit": self.a_inhibit,
            "a_safe": self.a_safe,
            "domain": self.domain,
        }


class SemanticRouter:
    """
    Evaluates runtime user prompts against stacked trigger embedding matrix E.
    """

    def __init__(self, embedder: Optional[SemanticTriggerEmbedder] = None, default_tau: float = 0.35, tau: Optional[float] = None):
        self.embedder = embedder or SemanticTriggerEmbedder()
        self.default_tau = tau if tau is not None else default_tau
        self.rules: List[ReflexRule] = []
        self._matrix: Optional[np.ndarray] = None  # (K, D)

    def clear(self) -> None:
        self.rules.clear()
        self._matrix = None

    def add_rule(
        self,
        scenario_id: Any,
        trigger_prompt: str = "",
        a_inhibit: str = "",
        a_safe: str = "",
        v_inhibit: Optional[List[int]] = None,
        domain: str = "",
    ) -> ReflexRule:
        """Embeds trigger prompt and registers rule in buffer, or accepts ReflexRule instance."""
        if hasattr(scenario_id, "trigger_pattern") or hasattr(scenario_id, "trigger_prompt"):
            # ReflexRule instance passed
            r = scenario_id
            scen_id = getattr(r, "scenario_id", getattr(r, "rule_id", "rule"))
            trig = getattr(r, "trigger_prompt", getattr(r, "trigger_pattern", ""))
            emb = getattr(r, "embedding", None)
            if emb is None:
                emb = self.embedder.embed(trig)
            inh = getattr(r, "a_inhibit", getattr(r, "inhibited_action", ""))
            safe = getattr(r, "a_safe", getattr(r, "safe_alternative", ""))
            dom = getattr(r, "domain", "")
            vin = getattr(r, "v_inhibit", getattr(r, "inhibited_tokens", []))
            
            rule = ReflexRule(
                scenario_id=scen_id,
                trigger_prompt=trig,
                embedding=emb,
                a_inhibit=inh,
                a_safe=safe,
                v_inhibit=vin,
                domain=dom,
            )
            rule.rule_id = scen_id
            self.rules.append(rule)
            self._matrix = None
            return rule

        emb = self.embedder.embed(trigger_prompt)
        rule = ReflexRule(
            scenario_id=scenario_id,
            trigger_prompt=trigger_prompt,
            embedding=emb,
            a_inhibit=a_inhibit,
            a_safe=a_safe,
            v_inhibit=v_inhibit,
            domain=domain,
        )
        rule.rule_id = str(scenario_id)
        self.rules.append(rule)
        self._matrix = None  # Invalidate cached matrix
        return rule

    def _ensure_matrix(self) -> np.ndarray:
        if self._matrix is None:
            if not self.rules:
                self._matrix = np.empty((0, 384), dtype=np.float32)
            else:
                self._matrix = np.stack([r.embedding for r in self.rules], axis=0)
        return self._matrix

    def route(
        self,
        query: str,
        tau: Optional[float] = None,
    ) -> Tuple[float, bool, Optional[ReflexRule]]:
        """
        Calculates s_t = max_{r_i in R} (e_t^T e_CS,i).
        Returns: (router_score, gate_activated, winning_rule)
        """
        threshold = self.default_tau if tau is None else tau
        if not self.rules:
            return 0.0, False, None

        q_emb = self.embedder.embed(query)  # (384,)
        mat = self._ensure_matrix()  # (K, 384)

        # Dot product with normalized vectors equals cosine similarity
        scores = np.dot(mat, q_emb)  # (K,)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        # Exact gate decision: fire if and only if router_score >= tau
        gate_activated = best_score >= threshold
        winning_rule = self.rules[best_idx]

        return best_score, gate_activated, winning_rule


# Alias for backward compatibility with test suites
SemanticReflexRouter = SemanticRouter

