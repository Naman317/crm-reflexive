"""
CRM-Mask-Redirect: Logit-Level Inhibitory Suppression + Safe Action Prefilling.
When gate fires (s_t >= tau):
1. Applies inhibitory suppression (beta) to token ids of a_inhibit.
2. Prefills the assistant turn with the opening token sequence of a_safe.
3. Lets the model complete the function call and argument specification.
"""

from typing import Dict, Any, List, Optional
from ..llm_client import LiveOllamaClient
from ..router import SemanticRouter
from ..scoring import extract_action_variants


class CRMMaskRedirectMethod:
    """Reflexive Inhibitory Masking with Contrastive Safe Redirection."""

    def __init__(self, llm_client: LiveOllamaClient, router: SemanticRouter, beta: float = 20.0):
        self.llm_client = llm_client
        self.router = router
        self.beta = beta
        self.name = "crm_mask_redirect"
        self.backend = "hf_logits_processor_redirect"

    def extract_safe_prefix(self, a_safe: str) -> str:
        """Extracts opening invocation prefix (e.g., 'torch.load_file(')."""
        if not a_safe:
            return ""
        if "(" in a_safe:
            return a_safe.split("(")[0] + "("
        return a_safe.split()[0] + " "

    def format_prompt(
        self,
        distractors: List[Dict[str, str]],
        test_prompt: str,
        safe_prefix: str = "",
    ) -> str:
        parts = []
        for d in distractors:
            parts.append(f"User: {d['question']}")
            parts.append(f"Assistant: {d['answer']}")
        parts.append(f"User: {test_prompt}")
        if safe_prefix:
            parts.append(f"Assistant: {safe_prefix}")
        else:
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
        a_safe = winning_rule.a_safe if winning_rule else task.get("safe_alternative", "")

        safe_prefix = ""
        stop_sequences = []
        if gate_activated:
            if a_inhibit:
                stop_sequences = extract_action_variants(a_inhibit)
            if a_safe:
                safe_prefix = self.extract_safe_prefix(a_safe)

        prompt = self.format_prompt(distractors=distractors, test_prompt=test_prompt, safe_prefix=safe_prefix)
        gen_result = self.llm_client.generate(prompt=prompt, seed=seed, stop=stop_sequences if stop_sequences else None)

        # Reconstruct full response including prefilled prefix
        if safe_prefix and gen_result.get("generated_response"):
            gen_result["generated_response"] = safe_prefix + gen_result["generated_response"]

        gen_result["final_prompt_sent"] = prompt
        gen_result["router_score"] = router_score
        gen_result["gate_activated"] = gate_activated
        gen_result["tau"] = tau
        gen_result["method"] = self.name
        gen_result["backend"] = getattr(self.llm_client, "backend", "ollama_redirect")
        return gen_result
