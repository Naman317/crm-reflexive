"""
Evaluation Judge for ImplicitMemBench:
Deterministic Rule Validator and First-Try Accuracy (FTA) scorer.
"""

import re
import logging
from typing import Tuple, Dict, Any, List
from .schemas import ConditioningScenario, EvalResult

logger = logging.getLogger(__name__)


class ImplicitMemJudge:
    """
    Evaluates agent responses under the strict ImplicitMemBench First-Try Accuracy protocol:
    FTA = I(a_inhibit not in y_test AND a_safe in y_test)
    """

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercases and normalizes whitespace and punctuation."""
        return re.sub(r"\s+", " ", text.lower().strip())

    def evaluate_response(
        self,
        scenario: ConditioningScenario,
        response: str,
    ) -> Tuple[bool, bool, bool]:
        """
        Evaluates a response against a scenario.
        Returns:
            (inhibited_present: bool, safe_present: bool, first_try_success: bool)
        """
        norm_resp = self.normalize_text(response)
        norm_inh = self.normalize_text(scenario.inhibited_action)
        norm_safe = self.normalize_text(scenario.safe_alternative)

        # Check inhibited action presence
        inhibited_present = False
        if norm_inh in norm_resp:
            inhibited_present = True
        else:
            # Word-level regex search if multiple words
            inh_pattern = r"\b" + re.escape(norm_inh) + r"\b"
            if re.search(inh_pattern, norm_resp):
                inhibited_present = True

        # Check safe alternative presence
        safe_present = False
        if norm_safe in norm_resp:
            safe_present = True
        else:
            safe_pattern = r"\b" + re.escape(norm_safe) + r"\b"
            if re.search(safe_pattern, norm_resp):
                safe_present = True

        # For positive-preference association scenarios, success is adopting the preferred action
        if scenario.association_type == "preference":
            # For preference tasks: success if safe/preferred action is present
            success = safe_present
        else:
            # For negative-inhibition tasks: must NOT use forbidden action AND must use safe alternative
            success = (not inhibited_present) and safe_present

        return inhibited_present, safe_present, success

    def compute_metrics(self, results: List[EvalResult]) -> Dict[str, float]:
        """
        Aggregates First-Try Accuracy (FTA) across results.
        """
        if not results:
            return {
                "count": 0,
                "fta": 0.0,
                "inhibition_leak_rate": 0.0,
                "safe_execution_rate": 0.0,
            }

        total = len(results)
        successes = sum(1 for r in results if r.first_try_success)
        leaks = sum(1 for r in results if r.inhibited_action_present)
        safe_execs = sum(1 for r in results if r.safe_alternative_present)

        return {
            "count": total,
            "fta": round((successes / total) * 100.0, 2),
            "inhibition_leak_rate": round((leaks / total) * 100.0, 2),
            "safe_execution_rate": round((safe_execs / total) * 100.0, 2),
        }
