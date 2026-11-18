"""
Configuration loader for CRM evaluation harness.
Loads config.yaml strictly; no magic numbers in code.
"""

import os
from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class ModelConfig:
    name: str = "qwen2.5:3b"
    ollama_endpoint: str = "http://localhost:11434"
    temperature: float = 0.0
    num_predict: int = 128
    num_ctx: int = 10240
    timeout_seconds: float = 600.0
    seeds: List[int] = field(default_factory=lambda: [42, 137, 2026])


@dataclass
class EmbedderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    device: str = "cpu"
    normalize: bool = True


@dataclass
class ParametersConfig:
    tau_default: float = 0.35
    tau_list: List[float] = field(default_factory=lambda: [0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55, 0.65])
    horizons: List[int] = field(default_factory=lambda: [4, 12, 30, 120])
    mask_beta: float = 20.0
    rag_top_k: int = 3
    winning_prompt_candidate: str = "Candidate_3_ExecutionRule"
    directive_template_inhibit: str = "Execution Rule: Avoid {a_inhibit}. Call {a_safe} directly.\n"
    directive_template_pref: str = "Execution Rule: Call {a_safe} directly.\n"


@dataclass
class PathsConfig:
    data_dir: str = "data"
    results_dir: str = "results"
    raw_logs_dir: str = "results/raw_logs"
    figures_dir: str = "results/figures"
    tables_dir: str = "results/tables"


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    parameters: ParametersConfig = field(default_factory=ParametersConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Loads configuration from yaml file."""
    if not os.path.exists(config_path):
        # Also check root if running from within src or scripts
        alt_path = os.path.join(os.path.dirname(__file__), "..", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig(
            model=ModelConfig(**raw.get("model", {})),
            embedder=EmbedderConfig(**raw.get("embedder", {})),
            parameters=ParametersConfig(**raw.get("parameters", {})),
            paths=PathsConfig(**raw.get("paths", {})),
        )
    return AppConfig()
