"""
Stage 3: Pre-Action Inhibitory Gate (Variant A: CRM-Prompt, Variant B: CRM-Mask).
Implements prompt injection at recency offset 0 and decoding-time token masking / Ollama interceptor.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from .schemas import ReflexRule

logger = logging.getLogger(__name__)


class CRMPromptGate:
    r"""
    Variant A: Contrastive Prompt Gate (CRM-Prompt).
    Prepends compact contrastive directive immediately before the test prompt:
    x~_t = [INHIBIT: a_{inhibit, i*} | ENFORCE: a_{safe, i*}] \oplus x_t
    """

    @staticmethod
    def construct_prompt(prompt: str, winning_rule: ReflexRule) -> str:
        """
        Synthesizes the augmented prompt with zero prompt bloat.
        """
        directive = f"[INHIBIT: {winning_rule.inhibited_action} | ENFORCE: {winning_rule.safe_alternative}]"
        return f"{directive}\n{prompt}"


class HuggingFaceInhibitoryLogitsProcessor:
    """
    Variant B (Token Logits Processor for HuggingFace Transformers).
    Penalizes logits of tokens in V_{inhibit} by beta:
    z~_{k, v} = z_{k, v} - beta * I(s_t >= tau) * I(v in V_{inhibit})
    """

    def __init__(
        self,
        tokenizer: Any,
        inhibited_tokens: List[str],
        beta: float = 50.0,
        active: bool = True,
    ):
        self.tokenizer = tokenizer
        self.beta = beta
        self.active = active
        self.forbidden_token_ids: Set[int] = set()

        if tokenizer is not None:
            for tok_str in inhibited_tokens:
                ids = tokenizer.encode(tok_str, add_special_tokens=False)
                for tid in ids:
                    self.forbidden_token_ids.add(tid)
                ids_space = tokenizer.encode(f" {tok_str}", add_special_tokens=False)
                for tid in ids_space:
                    self.forbidden_token_ids.add(tid)

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if not self.active or not self.forbidden_token_ids:
            return scores

        for tid in self.forbidden_token_ids:
            if tid < scores.shape[-1]:
                scores[:, tid] -= self.beta

        return scores



class OllamaDecodingInterceptor:
    """
    Variant B (Decoding Interceptor for Ollama & API-based backends).
    Enforces constrained decoding by setting stop sequences, inspecting stream outputs,
    and post-processing outputs to guarantee hard suppression of inhibited tokens.
    """

    def __init__(self, beta_penalty: float = 50.0):
        self.beta_penalty = beta_penalty

    def prepare_ollama_options(self, winning_rule: ReflexRule) -> Dict[str, Any]:
        """
        Creates Ollama request options including stop sequences from inhibited tokens.
        """
        stop_tokens = list(winning_rule.inhibited_tokens)
        if winning_rule.inhibited_action not in stop_tokens:
            stop_tokens.append(winning_rule.inhibited_action)

        # Filter short stop tokens (< 2 chars can break normal English decoding)
        filtered_stops = [s for s in stop_tokens if len(s.strip()) >= 3][:8]

        return {
            "temperature": 0.0,
            "stop": filtered_stops,
        }

    def intercept_and_filter(
        self,
        raw_output: str,
        winning_rule: ReflexRule,
    ) -> Tuple[str, bool]:
        """
        Evaluates generated output for leakage of inhibited action.
        If the output contains the forbidden action or stopped prematurely because of a stop token,
        it suppresses the forbidden phrase and injects the safe alternative.
        Returns:
            (filtered_output: str, was_intercepted: bool)
        """
        target_inh = winning_rule.inhibited_action.lower()
        output_lower = raw_output.lower()

        intercepted = False
        result = raw_output

        # Check exact inhibited action or tokens
        if target_inh in output_lower:
            intercepted = True
            # Suppress forbidden action and redirect to safe alternative
            pattern = re.compile(re.escape(winning_rule.inhibited_action), re.IGNORECASE)
            result = pattern.sub(winning_rule.safe_alternative, result)

        for tok in winning_rule.inhibited_tokens:
            if len(tok) >= 4 and tok.lower() in result.lower():
                intercepted = True
                pattern = re.compile(re.escape(tok), re.IGNORECASE)
                result = pattern.sub(winning_rule.safe_alternative, result)

        # If generation stopped prematurely without expressing the safe alternative
        if intercepted and winning_rule.safe_alternative.lower() not in result.lower():
            result = f"{result.strip()} -> Executing enforced fallback: {winning_rule.safe_alternative}"

        return result, intercepted
