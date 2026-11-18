"""
Vanilla Baseline Method.
Conversation context = learning turn + M real distractor turns + test prompt.
Zero-shot execution with no external memory intervention.
"""

from typing import Dict, Any, List
from ..llm_client import LiveOllamaClient


class VanillaMethod:
    """Vanilla generation baseline with real distractor context expansion."""

    def __init__(self, llm_client: LiveOllamaClient):
        self.llm_client = llm_client
        self.name = "vanilla"
        self.backend = "ollama"

    def format_prompt(
        self,
        learning_prompt: str,
        a_inhibit: str,
        u_us: str,
        a_safe: str,
        distractors: List[Dict[str, str]],
        test_prompt: str,
    ) -> str:
        parts = []

        # 1. Learning Phase turn
        parts.append(f"User: {learning_prompt}")
        if a_inhibit:
            parts.append(f"Assistant: Executing action: {a_inhibit}")
            parts.append(f"Environment: Feedback: {u_us} Required action: {a_safe}")
        else:
            parts.append(f"Assistant: Executing preferred format: {a_safe}")

        # 2. Real distractor turns
        for d in distractors:
            parts.append(f"User: {d['question']}")
            parts.append(f"Assistant: {d['answer']}")

        # 3. Test prompt
        parts.append(f"User: {test_prompt}")
        parts.append("Assistant:")

        return "\n\n".join(parts)

    def run_trial(
        self,
        task: Dict[str, Any],
        distractors: List[Dict[str, str]],
        test_prompt: str,
        seed: int = 42,
    ) -> Dict[str, Any]:
        prompt = self.format_prompt(
            learning_prompt=task.get("learning_prompt", task.get("test_probe", "")),
            a_inhibit=task.get("inhibited_action", ""),
            u_us=task.get("feedback_text", task.get("metadata", {}).get("error_type", "Execution failed.")),
            a_safe=task.get("safe_alternative", ""),
            distractors=distractors,
            test_prompt=test_prompt,
        )

        gen_result = self.llm_client.generate(prompt=prompt, seed=seed)
        gen_result["final_prompt_sent"] = prompt
        gen_result["router_score"] = 0.0
        gen_result["gate_activated"] = False
        gen_result["tau"] = 0.35
        gen_result["method"] = self.name
        gen_result["backend"] = self.backend

        return gen_result
