"""
CRM-Mask (Variant B): Token-Level Decoding Interception & Logit Suppression.
When gate fires (s_t >= tau), applies inhibitory suppression to V_inhibit tokens.
Explicitly labels backend in trial metadata (e.g. 'ollama_stop_interception' vs 'hf_logits_processor').
"""

from typing import Dict, Any, List, Optional
from ..llm_client import LiveOllamaClient
from ..router import SemanticRouter
from ..scoring import extract_action_variants


class CRMMaskMethod:
    """Inhibitory Token Masking & Decoding Interception Method."""

    def __init__(self, llm_client: LiveOllamaClient, router: SemanticRouter, beta: float = 20.0):
        self.llm_client = llm_client
        self.router = router
        self.beta = beta
        self.name = "crm_mask"
        self.backend = "ollama_stop_interception"

    def format_prompt(
        self,
        distractors: List[Dict[str, str]],
        test_prompt: str,
    ) -> str:
        parts = []
        for d in distractors:
            parts.append(f"User: {d['question']}")
            parts.append(f"Assistant: {d['answer']}")
        parts.append(f"User: {test_prompt}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def run_trial(
        self,
        task: Dict[str, Any],
        distractors: List[Dict[str, str]],
        test_prompt: str,
        tau: float = 0.35,
        seed: int = 42,
    ) -> Dict[str, Any]:
        router_score, gate_activated, winning_rule = self.router.route(test_prompt, tau=tau)

        a_inhibit = winning_rule.a_inhibit if winning_rule else task.get("inhibited_action", "")
        prompt = self.format_prompt(distractors=distractors, test_prompt=test_prompt)

        stop_sequences = []
        if gate_activated and a_inhibit:
            # Extract common stop variants for the forbidden token sequence
            stop_sequences = extract_action_variants(a_inhibit)

        gen_result = self.llm_client.generate(prompt=prompt, seed=seed, stop=stop_sequences if stop_sequences else None)
        gen_result["final_prompt_sent"] = prompt
        gen_result["router_score"] = router_score
        gen_result["gate_activated"] = gate_activated
        gen_result["tau"] = tau
        gen_result["method"] = self.name
        gen_result["backend"] = self.backend
        gen_result["stop_sequences_applied"] = stop_sequences
        gen_result["winning_rule"] = winning_rule.to_dict() if winning_rule else None

        return gen_result
