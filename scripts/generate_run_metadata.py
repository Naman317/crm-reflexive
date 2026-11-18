import hashlib
import json
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(root_dir, "config.yaml")

with open(config_path, "rb") as f:
    cfg_bytes = f.read()
cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()

meta = {
    "config_sha256": cfg_hash,
    "git_commit": "unversioned_local",
    "python_version": sys.version,
    "torch_version": "2.10.0+cpu",
    "transformers_version": "5.17.0",
    "model_name": "qwen2.5:3b",
    "model_digest": "357c53fb659c",
    "seeds": [42, 123, 999],
    "parameters": {
        "tau_default": 0.35,
        "mask_beta": 20.0,
        "rag_top_k": 3,
        "winning_prompt_candidate": "Candidate_1_Direct_Natural",
        "directive_template_inhibit": "Do not use {a_inhibit}. Use {a_safe} instead.\n",
        "directive_template_pref": "Use {a_safe}.\n"
    },
    "backend_per_method": {
        "vanilla": "ollama",
        "mem0_rag": "ollama",
        "crm_prompt": "ollama",
        "crm_mask": "ollama_stop_interception",
        "crm_mask_redirect": "hf_logits_processor"
    }
}

out_path = os.path.join(root_dir, "results", "run_metadata.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Config SHA-256: {cfg_hash}")
print(f"Saved run metadata to {out_path}")
