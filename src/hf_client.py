"""
Hugging Face Transformers Client for GPU & Matched Baseline Comparison.
Supports:
- Qwen/Qwen2.5-3B-Instruct and Qwen/Qwen2.5-7B-Instruct (4-bit via bitsandbytes)
- Native PyTorch LogitsProcessor for real inhibitory token masking
- Exact prompt token and generated token accounting
"""

import time
import torch
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList


class InhibitoryMaskLogitsProcessor(LogitsProcessor):
    """
    Subtracts beta from the token ids corresponding to forbidden action a_inhibit.
    """
    def __init__(self, inhibited_token_ids: List[int], beta: float = 20.0):
        self.inhibited_token_ids = inhibited_token_ids
        self.beta = beta

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for tid in self.inhibited_token_ids:
            scores[:, tid] -= self.beta
        return scores


class HFTransformersClient:
    """
    Direct Hugging Face transformers execution client.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit: bool = False,
        temperature: float = 0.0,
        max_new_tokens: int = 48,
    ):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.backend = "transformers"

        print(f"Loading HF model {model_name} on {device} (4-bit: {load_in_4bit})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        kwargs = {"low_cpu_mem_usage": True}
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["device_map"] = "auto"
        elif device == "cuda":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not load_in_4bit and device != "auto":
            self.model.to(device)
        self.model.eval()
        print(f"HF Model {model_name} successfully initialized!")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        seed: int = 42,
        stop: Optional[List[str]] = None,
        inhibited_tokens: Optional[List[str]] = None,
        beta: float = 20.0,
    ) -> Dict[str, Any]:
        torch.manual_seed(seed)
        full_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        # Setup logits processors if inhibited tokens provided
        processors = LogitsProcessorList()
        if inhibited_tokens:
            inhibit_ids = set()
            for token_str in inhibited_tokens:
                # Tokenize with and without leading space, quotes, backticks
                for variant in [token_str, f" {token_str}", f"'{token_str}'", f"`{token_str}`"]:
                    ids = self.tokenizer.encode(variant, add_special_tokens=False)
                    inhibit_ids.update(ids)
            if inhibit_ids:
                processors.append(InhibitoryMaskLogitsProcessor(list(inhibit_ids), beta=beta))

        start_time = time.perf_counter()
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False if self.temperature == 0.0 else True,
                    temperature=self.temperature if self.temperature > 0.0 else None,
                    logits_processor=processors if len(processors) > 0 else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            generated_ids = out[0][prompt_tokens:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            eval_tokens = len(generated_ids)

            # Check stop sequences
            if stop:
                for s in stop:
                    if s in generated_text:
                        generated_text = generated_text.split(s)[0].strip()

            return {
                "generated_response": generated_text,
                "prompt_tokens": prompt_tokens,
                "eval_tokens": eval_tokens,
                "latency_ms": latency_ms,
                "error_message": None,
                "model": self.model_name,
                "seed": seed,
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "generated_response": "",
                "prompt_tokens": prompt_tokens,
                "eval_tokens": 0,
                "latency_ms": latency_ms,
                "error_message": str(e),
                "model": self.model_name,
                "seed": seed,
            }
