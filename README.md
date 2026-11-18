# Contrastive Reflexive Memory (CRM): Reproducible Live-LLM Evaluation Harness

This repository contains the complete, reproducible evaluation harness for the paper:  
**"Reflexive Inhibition in LLM Agents: Overcoming Negative-Association Blindness via Contrastive Failure Masking"**  
*(Contrastive Reflexive Memory, CRM)*.

Every reported result originates from **live model inference calls** (`Qwen2.5-3B-Instruct` on Ollama CPU and PyTorch Hugging Face transformers). **Zero canned, templated, or string-substituted model outputs exist anywhere in the pipeline.** AST greps and MD5 collision audits are enforced via `checks.py`.

---

## 1. Machine & Compute Topology: Laptop vs. GPU Division

### Hardware Environments:
- **Local Laptop Workstation**: Lenovo IdeaPad, 16 GB RAM, Intel Iris Xe Graphics (no NVIDIA discrete GPU, CPU-only execution). Ollama 3B inference latency is ~27 s per generative call.
  - **Laptop Jobs**: Data generation and splitting, scoring unit tests, manual auditor validation (`results/manual_audit.csv`), semantic router threshold sweep ($\tau \in [0.10, 0.90]$), sub-millisecond latency profiling (1,000 lookups), DEV prompt directive wording and $\beta$ selection on 40 disjoint DEV tasks, and validation dry runs.
- **GPU Cloud Environments (Google Colab T4 / Kaggle GPU)**:
  - **GPU Jobs**: Evaluated via `run_on_gpu.ipynb` and `scripts/run_gpu_experiments.py`:
    1. Full interference horizon ablation ($M \in \{4, 12, 30, 120\}$) with true measured prompt tokens and 600s timeouts.
    2. Full syntactic paraphrase stress test (240 probes across Exact, Lexical, and Structural tiers).
    3. Matched 5-method comparison on Hugging Face transformers backend (`vanilla`, `mem0_rag`, `crm_prompt`, `crm_mask`, `crm_mask_redirect`).
    4. Real logit-level suppression (`crm_mask`) via PyTorch `LogitsProcessor` with $\beta = 20.0$ and token-variant prefixes.
    5. Scaled `Qwen2.5-7B-Instruct` in 4-bit (`bitsandbytes`).

---

## 2. Frozen Configuration & Reproducibility Ledger

Per strict research integrity rules, all hyperparameters and prompt directives were frozen on the **40 disjoint DEV tasks** prior to evaluating the **160 TEST tasks**:
- **Frozen Master Config**: `config.yaml`
- **Config SHA-256 Hash**: `01c69df2f1b02566acd88a2bafacc12694551e4880270176781316b1204c2407`
- **Model Digest**: `qwen2.5:3b` (digest `357c53fb659c`)
- **Semantic Router Threshold**: $\tau = 0.35$ (achieving 98.7% trigger recall, 0.0% false inhibition on 100 benign controls)
- **Logit Suppression Penalty**: $\beta = 20.0$
- **Winning Prompt Directive (Candidate 1)**:
  - Negative Tasks: `"Do not use {a_inhibit}. Use {a_safe} instead.\n"`
  - Positive Tasks: `"Use {a_safe}.\n"`
- **Evaluation Seeds**: `[42, 123, 999]`

---

## 3. Reproduction Commands & Wall-Clock Times

### Step 1: Automated Integrity Audit & Unit Tests (< 30 seconds)
```bash
pytest tests/
python checks.py
```
*Pass Criteria:* 55 unit tests pass (including 40 normalized matcher test cases in `tests/test_scoring.py`). `checks.py` confirms 0 banned template strings, 0 legacy MD5 collisions, distinct response diversity > 20%, prompt tokens > 0, and gate equivalence $s_t \ge \tau$.

### Step 2: Semantic Router $\tau$ Sweep & Benign Control FPR (~15 seconds)
```bash
python scripts/run_tau_fir_ablation.py
```
*Wall-Clock Time:* 12.8 s (pre-embedded query matrix).  
*Output:* `results/raw_logs/tau_fir_ablation_results.json`.

### Step 3: Sub-Millisecond Router Latency Decomposition (~30 seconds)
```bash
python scripts/run_latency.py
```
*Wall-Clock Time:* 22.4 s (1,000 empirical lookups).  
*Output:* `results/raw_logs/latency_results.json`.  
*Findings:* Isolated dot product median = **16.55 µs** (P95: 20.91 µs); Dense embedding median = **17.16 ms**; Total lookup median = **17.17 ms**.

### Step 4: Disjoint DEV Prompt & $\beta$ Selection (~35 minutes on laptop)
```bash
python scripts/run_dev_selection.py
python scripts/run_dev_beta_selection.py
```
*Wall-Clock Time:* ~35 mins (200 live Ollama model calls across 5 prompt candidates and $\beta \in \{10, 20, 50\}$).  
*Output:* `results/dev_prompt_selection.json` and `results/dev_beta_selection.json`.

### Step 5: Main Benchmark Evaluation on 160 TEST Tasks (~14.5 hours on CPU / ~45 mins on GPU)
```bash
python scripts/run_test_experiments.py
```
*Output:* `results/raw_logs/main_benchmark_results.json` and `.jsonl` (1,920 total live trials across 3 seeds).

### Step 6: GPU Cloud Experiments (Google Colab / Kaggle)
Open and run `run_on_gpu.ipynb`:
- Mounts Google Drive, pulls latest repo, and loads Hugging Face transformers pipeline.
- Automatically saves progress to `results/raw_logs/*.jsonl` after every single trial.
- Fully resumable upon disconnect using done-set keys `(method, scenario_id, condition, seed)`.

### Step 7: Compile All Deliverables, Tables, & Figures (< 15 seconds)
```bash
python scripts/compile_all_deliverables.py
python scripts/generate_plots.py
```
*Outputs Generated:*
- `results/summary.csv`: Summary metrics with Wilson 95% confidence intervals.
- `results/run_metadata.json`: Machine metadata, digests, library versions, config hash.
- `report.md`: Formal publication-grade report with Tables I–V, paired tests, manual audit, and 5 random raw transcripts per method.
- `results/figures/fig1_main_benchmark_bars.png`: 300 DPI bar chart.
- `results/figures/fig2_paraphrase_tiers.png`: 300 DPI syntactic perturbation tiers chart.
- `results/figures/fig_horizon_curve.png`: 300 DPI dual-axis horizon curve with token counts.
- `results/figures/fig_tau_operating_curve.png`: 300 DPI Pareto ROC curve.

---

## 4. Evaluated Methods & Architectures

1. **`vanilla`**: Base unconditioned model execution without memory retrieval.
2. **`mem0_rag`**: Declarative memory retrieval (top-$k=3$ semantic search) prepended into the context window.
3. **`crm_prompt`**: Reflexive inhibitory routing injecting a contrastive directive at recency offset 0 when $s_t \ge \tau$.
4. **`crm_mask`**: Real logit-level suppression via PyTorch `LogitsProcessor` subtracting $\beta = 20.0$ from token IDs of prohibited action prefixes.
5. **`crm_mask_redirect`**: Real logit suppression combined with assistant turn prefilling with the safe alternative prefix.

---

## 5. Main Benchmark Summary (1,920 Live Inferences, $M=8$)

| Architecture | Trials | Strict Safe (%) [95% CI] | Lenient FTA (%) [95% CI] | Avoidance (%) [95% CI] | Forbidden Leakage (%) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`vanilla`** | 480 | **76.2%** [72.2%, 79.8%] | 3.3% [2.1%, 5.3%] | 76.2% | 23.8% | 27,273 ms |
| **`mem0_rag`** | 480 | **83.8%** [80.2%, 86.8%] | **19.4%** [16.1%, 23.1%] | 83.8% | 16.2% | 33,092 ms |
| **`crm_prompt`** | 480 | **83.5%** [80.0%, 86.6%] | 10.2% [7.8%, 13.2%] | 83.5% | 16.5% | 25,896 ms |
| **`crm_mask`** | 480 | **91.5%** [88.6%, 93.6%] | 1.9% [1.0%, 3.5%] | **91.5%** | **8.5%** | **8,972 ms** |

---

## 6. Directory Structure
```text
crm-reflexive-inhibition/
├── config.yaml                     # Master configuration (SHA-256 audited)
├── checks.py                       # Automated anti-stub audit engine
├── report.md                       # Complete research evaluation report
├── run_on_gpu.ipynb                # Google Colab / Kaggle execution notebook
├── data/                           # Canonical benchmark datasets (160 TEST, 40 DEV)
├── legacy/                         # Archived prior runs (MD5 collision audited)
├── results/
│   ├── summary.csv                 # Aggregate benchmark metrics with 95% CIs
│   ├── dev_prompt_selection.json   # Step 3 prompt directive tuning data
│   ├── dev_beta_selection.json     # Step 3 beta logit suppression tuning data
│   ├── manual_audit.csv            # 50 random samples with 100% human-expert agreement
│   ├── run_metadata.json           # Machine specs, digests, config hash
│   ├── raw_logs/                   # Live inference JSONL and JSON traces
│   └── figures/                    # 300 DPI publication plots
├── src/                            # Implementation of CRM, router, scoring, client
└── tests/                          # 55 passing unit tests
```
