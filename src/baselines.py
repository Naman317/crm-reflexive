"""
Baseline Memory Architectures for ImplicitMemBench:
1. Vanilla LLM (Direct generation without memory intervention)
2. Explicit RAG Memory (Mem0 / MemGPT style passive memory retrieval)
3. Self-Reflection Baseline (Reflexion-style retrospective scratchpad)
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from .schemas import ConditioningScenario, ReflexRule
from .embedder import SemanticTriggerEmbedder

logger = logging.getLogger(__name__)


class VanillaBaseline:
    """
    Standard generation: context includes learning turns + interference turns + test prompt.
    """

    @staticmethod
    def format_prompt(scenario: ConditioningScenario, test_prompt: str) -> str:
        prompt_parts = []
        # Learning phase
        for turn in scenario.learning_turns:
            role = turn.get("role", "user").capitalize()
            prompt_parts.append(f"{role}: {turn.get('content', '')}")

        # Distractor turns
        for turn in scenario.interference_turns:
            role = turn.get("role", "user").capitalize()
            prompt_parts.append(f"{role}: {turn.get('content', '')}")

        # Test prompt
        prompt_parts.append(f"User: {test_prompt}")
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)


class ExplicitRAGMem0Baseline:
    """
    Mem0-style Explicit RAG Memory:
    Extracts declarative memory facts during the learning phase, indexes them in a dense index,
    and retrieves top-k facts to prepend as passive context.
    """

    def __init__(self, embedder: Optional[SemanticTriggerEmbedder] = None, top_k: int = 2):
        self.embedder = embedder or SemanticTriggerEmbedder()
        self.top_k = top_k
        self.memory_store: List[Dict[str, Any]] = []

    def clear(self) -> None:
        self.memory_store.clear()

    def index_learning_phase(self, scenario: ConditioningScenario) -> None:
        """
        Extracts declarative fact representations from learning turns.
        """
        for turn in scenario.learning_turns:
            content = turn.get("content", "")
            if turn.get("role") in ["environment", "user"] and len(content) > 10:
                emb = self.embedder.embed(content)
                self.memory_store.append({
                    "text": f"Observed guideline: {content}",
                    "embedding": emb,
                    "scenario_id": scenario.scenario_id,
                })

        # Also store canonical constraint summary as Mem0 would
        summary_text = (
            f"User experience note: For task regarding '{scenario.domain}', "
            f"do not use '{scenario.inhibited_action}' due to past failure. "
            f"Instead, use '{scenario.safe_alternative}'."
        )
        self.memory_store.append({
            "text": summary_text,
            "embedding": self.embedder.embed(summary_text),
            "scenario_id": scenario.scenario_id,
        })

    def retrieve_relevant_memories(self, query: str) -> List[str]:
        """
        Retrieves top-k memory strings based on dense cosine similarity.
        """
        if not self.memory_store:
            return []

        q_emb = self.embedder.embed(query)
        scored = []
        for mem in self.memory_store:
            sim = float(np.dot(q_emb, mem["embedding"]))
            scored.append((sim, mem["text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:self.top_k]]

    def format_prompt(self, scenario: ConditioningScenario, test_prompt: str) -> str:
        summary_text = (
            f"User experience note: For task regarding '{scenario.domain}', "
            f"do not use '{scenario.inhibited_action}' due to past failure. "
            f"Instead, use '{scenario.safe_alternative}'."
        )
        mem_block = f"### Retrieved Relevant Memories:\n- {summary_text}\n\n"

        prompt_parts = []
        # Distractor turns
        for turn in scenario.interference_turns:
            role = turn.get("role", "user").capitalize()
            prompt_parts.append(f"{role}: {turn.get('content', '')}")

        prompt_parts.append(f"User: {test_prompt}")
        prompt_parts.append("Assistant:")
        body = "\n\n".join(prompt_parts)

        return f"{mem_block}{body}"


class SelfReflectionBaseline:
    """
    Self-reflection baseline: prompts the agent to critique and verify against prior mistakes
    before producing the final execution command.
    """

    @staticmethod
    def format_prompt(scenario: ConditioningScenario, test_prompt: str) -> str:
        vanilla_prompt = VanillaBaseline.format_prompt(scenario, test_prompt)
        reflection_instruction = (
            "\n[Self-Reflection Instruction]: Before answering, pause and review earlier conversational "
            "turns for any past execution failures, errors, or penalized behaviors. State what must be avoided, "
            "then provide your verified action."
        )
        return f"{vanilla_prompt}{reflection_instruction}"
