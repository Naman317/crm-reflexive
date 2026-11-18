"""
Mem0-style Explicit RAG Memory Baseline.
Stores declarative memory notes extracted from learning feedback in a dense vector store.
At test time, retrieves top-k=3 notes by cosine similarity and prepends them as passive context.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from ..llm_client import LiveOllamaClient
from ..embedder import SemanticTriggerEmbedder


class Mem0RAGMethod:
    """Explicit RAG Baseline retrieving declarative notes into context."""

    def __init__(self, llm_client: LiveOllamaClient, embedder: Optional[SemanticTriggerEmbedder] = None, top_k: int = 3):
        self.llm_client = llm_client
        self.embedder = embedder or SemanticTriggerEmbedder()
        self.top_k = top_k
        self.name = "mem0_rag"
        self.backend = "ollama"
        self.memory_store: List[Dict[str, Any]] = []

    def clear(self) -> None:
        self.memory_store.clear()

    def index_task(self, task: Dict[str, Any]) -> None:
        """Indexes failure notes into dense memory store."""
        domain = task.get("domain", "system")
        a_inhibit = task.get("inhibited_action", "")
        a_safe = task.get("safe_alternative", "")
        u_us = task.get("feedback_text", task.get("metadata", {}).get("error_type", "Operation failed."))

        if a_inhibit:
            note_text = (
                f"Execution Rule for {domain}: Avoid {a_inhibit} due to failure ({u_us}). "
                f"Call {a_safe} directly."
            )
        else:
            note_text = (
                f"Execution Rule for {domain}: Call {a_safe} directly."
            )

        emb = self.embedder.embed(note_text)
        self.memory_store.append({
            "text": note_text,
            "embedding": emb,
            "scenario_id": task.get("scenario_id", ""),
        })

    def retrieve(self, query: str) -> List[str]:
        """Retrieves top-k notes by cosine similarity."""
        if not self.memory_store:
            return []

        q_emb = self.embedder.embed(query)
        scored = []
        for item in self.memory_store:
            sim = float(np.dot(q_emb, item["embedding"]))
            scored.append((sim, item["text"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:self.top_k]]

    def format_prompt(
        self,
        retrieved_notes: List[str],
        distractors: List[Dict[str, str]],
        test_prompt: str,
    ) -> str:
        parts = []

        if retrieved_notes:
            parts.append("Context notes from previous interactions:")
            for note in retrieved_notes:
                parts.append(f"- {note}")

        # Distractors
        for d in distractors:
            parts.append(f"User: {d['question']}")
            parts.append(f"Assistant: {d['answer']}")

        # Test prompt
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
        # Retrieve top-k notes
        retrieved_notes = self.retrieve(test_prompt)

        prompt = self.format_prompt(
            retrieved_notes=retrieved_notes,
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
        gen_result["retrieved_notes"] = retrieved_notes

        return gen_result
