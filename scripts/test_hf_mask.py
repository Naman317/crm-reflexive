import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_id = "Qwen/Qwen2.5-3B-Instruct"

class InhibitoryMaskLogitsProcessor(LogitsProcessor):
    def __init__(self, inhibited_token_ids: list[int], beta: float = 20.0):
        self.inhibited_token_ids = inhibited_token_ids
        self.beta = beta

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for tid in self.inhibited_token_ids:
            scores[:, tid] -= self.beta
        return scores

def benchmark_hf():
    print("Loading tokenizer and model from local HF cache...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    print(f"Loaded HF model in {time.perf_counter() - t0:.2f}s")

    prompt = "User: Run lint inspection on repository files.\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Inhibit token "subprocess"
    tok_ids = tokenizer.encode("subprocess", add_special_tokens=False)
    processor = InhibitoryMaskLogitsProcessor(inhibited_token_ids=tok_ids, beta=20.0)
    processors = LogitsProcessorList([processor])

    print("Timing 5 HF trials...")
    for i in range(5):
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=24, logits_processor=processors, do_sample=False)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"HF Trial {i+1}: {time.perf_counter() - t1:.2f}s | resp: {resp[:40]}...")

if __name__ == "__main__":
    benchmark_hf()
