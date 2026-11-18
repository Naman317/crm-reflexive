"""
CRM-Prompt (Variant A): Recency-Offset-0 Contrastive Prompt Gate.
Evaluates s_t = max(e_t^T e_CS) >= tau against stored reflex buffer.
If gate activates, prepends [INHIBIT: <a_inhibit> | ENFORCE: <a_safe>] directly before query.
If gate remains dormant, passes query unchanged.
"""

from typing import Dict, Any, List, Optional
from ..llm_client import LiveOllamaClient
from ..router import SemanticRouter


class CRMPromptMethod:
    """Pre-action Contrastive Prompt Gating Method."""

    def __init__(self, llm_client: LiveOllamaClient, router: SemanticRouter):
        self.llm_client = llm_client
        self.router = router
        self.name = "crm_prompt"
        self.backend = "ollama"

    def format_prompt(
        self,
        gate_activated: bool,
        a_inhibit: str,
        a_safe: str,
        distractors: List[Dict[str, str]],
        test_prompt: str,
    ) -> str:
        parts = []

        # Distractor conversational context
        for d in distractors:
            parts.append(f"User: {d['question']}")
            parts.append(f"Assistant: {d['answer']}")

        # Recency-offset-0 contrastive directive if gate fired
        if gate_activated:
            if a_inhibit:
                directive = f"Execution Rule: Avoid {a_inhibit}. Call {a_safe} directly.\n"
            else:
                directive = f"Execution Rule: Call {a_safe} directly.\n"
            parts.append(f"User: {directive}{test_prompt}")
        else:
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
        a_safe = winning_rule.a_safe if winning_rule else task.get("safe_alternative", "")

        prompt = self.format_prompt(
            gate_activated=gate_activated,
            a_inhibit=a_inhibit,
            a_safe=a_safe,
            distractors=distractors,
            test_prompt=test_prompt,
        )

        gen_result = self.llm_client.generate(prompt=prompt, seed=seed)
        gen_result["final_prompt_sent"] = prompt
        gen_result["router_score"] = router_score
        gen_result["gate_activated"] = gate_activated
        gen_result["tau"] = tau
        gen_result["method"] = self.name
        gen_result["backend"] = self.backend
        gen_result["winning_rule"] = winning_rule.to_dict() if winning_rule else None

        return gen_result
