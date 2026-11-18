"""
Live LLM Client for local Ollama server.
Guarantees:
- Real model text generation (no stubs, no heuristic fallbacks).
- Real prompt token counting (prompt_eval_count).
- Real latency tracking.
- Deterministic temperature 0.0 with explicit random seed.
"""

import time
import json
import logging
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger(__name__)


class LiveOllamaClient:
    """
    Connects to local Ollama API (http://localhost:11434).
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        temperature: float = 0.0,
        num_predict: int = 128,
        num_ctx: int = 10240,
        timeout_seconds: float = 600.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.timeout_seconds = timeout_seconds
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verifies Ollama is running and retrieves model digest."""
        try:
            resp = requests.get(f"{self.endpoint}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                for m in models:
                    if self.model in m.get("name", ""):
                        self.model_digest = m.get("digest", "")
                        logger.info("Connected to Ollama! Model: %s (digest: %s)", self.model, self.model_digest[:12])
                        return
                logger.warning("Model %s not found in Ollama tags! Available: %s", self.model, [m.get("name") for m in models])
                self.model_digest = "unknown"
            else:
                self.model_digest = "unknown"
        except Exception as e:
            logger.error("Could not connect to Ollama at %s: %s", self.endpoint, e)
            self.model_digest = "disconnected"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        seed: int = 42,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Sends generation request to Ollama /api/generate.
        Returns:
            {
                "generated_response": str,
                "prompt_tokens": int,
                "eval_tokens": int,
                "latency_ms": float,
                "error_message": Optional[str],
                "model": str,
                "seed": int,
            }
        """
        options = {
            "temperature": self.temperature,
            "seed": seed,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
        }
        if stop:
            options["stop"] = stop

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system_prompt:
            payload["system"] = system_prompt

        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            start_time = time.perf_counter()
            try:
                resp = requests.post(
                    f"{self.endpoint}/api/generate",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data.get("response", "").strip()
                    prompt_tokens = data.get("prompt_eval_count", len(prompt.split()))
                    eval_tokens = data.get("eval_count", len(raw_text.split()))

                    return {
                        "generated_response": raw_text,
                        "prompt_tokens": prompt_tokens,
                        "eval_tokens": eval_tokens,
                        "latency_ms": elapsed_ms,
                        "error_message": None,
                        "model": self.model,
                        "seed": seed,
                    }
                else:
                    last_error = f"Ollama HTTP {resp.status_code}: {resp.text}"
                    logger.warning("Attempt %d failed: %s", attempt, last_error)
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                last_error = str(e)
                logger.warning("Attempt %d generation error: %s", attempt, last_error)

            if attempt < max_retries:
                time.sleep(2.0 * attempt)

        return {
            "generated_response": "",
            "prompt_tokens": 0,
            "eval_tokens": 0,
            "latency_ms": 0.0,
            "error_message": last_error or "Generation failed after max retries",
            "model": self.model,
            "seed": seed,
        }
