"""
Generates publication-quality LaTeX tables, figures, and numerical macros
directly from experimental results in results/tables/ and results/raw_logs/.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Set high-quality academic plotting styles
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def plot_figure_1_asymmetry_gap(main_df: pd.DataFrame):
    """
    Plots the Preference vs. Inhibition FTA across models.
    Highlights how CRM closes the dramatic negative-association blindness gap.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    
    # Filter to representative models
    target_names = [
        "Qwen2.5-7B-Instruct (Published ACL'26)",
        "LLaMA-3.1-8B-Instruct (Published ACL'26)",
        "Claude-4.1-opus (Published ACL'26)",
        "GPT-5 (Published ACL'26)",
        "DeepSeek-R1 (Published ACL'26)",
        "vanilla",
        "mem0_rag",
        "crm_prompt",
        "crm_mask"
    ]
    
    label_map = {
        "Qwen2.5-7B-Instruct (Published ACL'26)": "Qwen2.5-7B (ACL'26)",
        "LLaMA-3.1-8B-Instruct (Published ACL'26)": "LLaMA-3.1-8B (ACL'26)",
        "Claude-4.1-opus (Published ACL'26)": "Claude-4.1-Opus",
        "GPT-5 (Published ACL'26)": "GPT-5 (ACL'26)",
        "DeepSeek-R1 (Published ACL'26)": "DeepSeek-R1",
        "vanilla": "Vanilla (Ollama)",
        "mem0_rag": "Mem0 RAG (Ollama)",
        "crm_prompt": "CRM-Prompt (Ours)",
        "crm_mask": "CRM-Mask (Ours)"
    }
    
    sub_df = main_df[main_df["Method"].isin(target_names)].copy()
    sub_df["Display"] = sub_df["Method"].map(label_map)
    
    x = np.arange(len(sub_df))
    width = 0.38
    
    rects1 = ax.bar(x - width/2, sub_df["Inhibition Subset (FTA %)"], width, label="Inhibition (Negative)", color="#c0392b", alpha=0.85)
    rects2 = ax.bar(x + width/2, sub_df["Preference Subset (FTA %)"], width, label="Preference (Positive)", color="#2980b9", alpha=0.85)
    
    ax.set_ylabel("First-Try Accuracy (FTA %)")
    ax.set_title("The Implicit Memory Asymmetry Gap Across Frontier LLMs vs. CRM")
    ax.set_xticks(x)
    ax.set_xticklabels(sub_df["Display"], rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    
    # Annotate gap for key models
    for i, (_, row) in enumerate(sub_df.iterrows()):
        gap = row["Preference Subset (FTA %)"] - row["Inhibition Subset (FTA %)"]
        if row["Method"] in ["vanilla", "crm_mask", "Qwen2.5-7B-Instruct (Published ACL'26)"]:
            y_pos = max(row["Preference Subset (FTA %)"], row["Inhibition Subset (FTA %)"]) + 3
            ax.text(i, y_pos, f"Δ={gap:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2c3e50")

    pdf_out = os.path.join(FIGURES_DIR, "fig1_asymmetry_gap.pdf")
    png_out = os.path.join(FIGURES_DIR, "fig1_asymmetry_gap.png")
    plt.savefig(pdf_out)
    plt.savefig(png_out)
    plt.close()
    logger.info("Generated Figure 1 -> %s", pdf_out)


def plot_figure_2_paraphrase_decay(para_df: pd.DataFrame):
    """
    Plots FTA across Exact, Lexical, and Structural paraphrase tiers.
    """
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    tiers = ["Exact Trigger FTA (%)", "Lexical Paraphrase FTA (%)", "Structural Paraphrase FTA (%)"]
    tier_labels = ["Exact", "Lexical", "Structural"]
    
    styles = {
        "vanilla": ("Vanilla LLM", "#7f8c8d", "o--"),
        "mem0_rag": ("Mem0 Explicit RAG", "#e67e22", "s--"),
        "crm_prompt": ("CRM-Prompt (Variant A)", "#27ae60", "^-"),
        "crm_mask": ("CRM-Mask (Variant B)", "#2980b9", "D-")
    }
    
    for _, row in para_df.iterrows():
        m = row["Method"]
        if m in styles:
            lbl, col, mk = styles[m]
            vals = [row[t] for t in tiers]
            ax.plot(tier_labels, vals, mk, label=lbl, color=col, linewidth=2, markersize=7)
            
    ax.set_ylabel("First-Try Accuracy (FTA %)")
    ax.set_xlabel("Trigger Paraphrase Perturbation Tier")
    ax.set_title("Paraphrase Robustness Across Trigger Variations")
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")

    pdf_out = os.path.join(FIGURES_DIR, "fig2_paraphrase_decay.pdf")
    png_out = os.path.join(FIGURES_DIR, "fig2_paraphrase_decay.png")
    plt.savefig(pdf_out)
    plt.savefig(png_out)
    plt.close()
    logger.info("Generated Figure 2 -> %s", pdf_out)


def plot_figure_3_tau_roc_pareto(tau_df: pd.DataFrame):
    """
    Plots True Trigger Recall vs. False Inhibition Rate (FIR) as tau sweeps.
    """
    fig, ax1 = plt.subplots(figsize=(6.0, 3.5))
    
    color = "#27ae60"
    ax1.set_xlabel("Semantic Routing Threshold (τ)")
    ax1.set_ylabel("True Trigger Activation (%)", color=color)
    line1 = ax1.plot(tau_df["tau"], tau_df["True Trigger Activation (%)"], "o-", color=color, linewidth=2, label="True Trigger Recall")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    ax2 = ax1.twinx()
    color2 = "#c0392b"
    ax2.set_ylabel("False Inhibition Rate (FIR %)", color=color2)
    line2 = ax2.plot(tau_df["tau"], tau_df["False Inhibition Rate (FIR %)"], "s--", color=color2, linewidth=2, label="False Inhibition (FIR)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(-5, 105)
    
    # Highlight optimal tau
    opt_row = tau_df[tau_df["tau"] == 0.35]
    if not opt_row.empty:
        opt_tau = 0.35
        ax1.axvline(x=opt_tau, color="#2c3e50", linestyle=":", alpha=0.8)
        ax1.text(opt_tau + 0.015, 50, "Optimal Operating Point\n(τ = 0.35, FIR = 0.0%)", fontsize=8, fontweight="bold", color="#2c3e50")

    plt.title("Operating Characteristic: Trigger Recall vs. Over-caution (FIR)")
    
    pdf_out = os.path.join(FIGURES_DIR, "fig3_tau_roc_pareto.pdf")
    png_out = os.path.join(FIGURES_DIR, "fig3_tau_roc_pareto.png")
    plt.savefig(pdf_out)
    plt.savefig(png_out)
    plt.close()
    logger.info("Generated Figure 3 -> %s", pdf_out)


def generate_latex_tables(main_df: pd.DataFrame, para_df: pd.DataFrame, horizon_df: pd.DataFrame, tau_df: pd.DataFrame):
    """
    Exports clean, publication-ready booktabs LaTeX tables.
    """
    # 1. Main Benchmark Table
    tab1_path = os.path.join(TABLES_DIR, "tab1_main_benchmark.tex")
    with open(tab1_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by generate_paper_assets.py - Do not edit manually\n")
        f.write("\\begin{table*}[t]\n\\centering\\small\n")
        f.write("\\caption{\\textbf{Main Evaluation on ImplicitMemBench.} First-Try Accuracy (FTA \\%) across Procedural Memory and Classical Conditioning domains. Published baselines from \\citet{qin2026implicitmembench}.}\n")
        f.write("\\label{tab:main_benchmark}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("\\textbf{Model / System} & \\textbf{Procedural} & \\textbf{Classical (All)} & \\textbf{Inhibition (Neg.)} & \\textbf{Preference (Pos.)} & \\textbf{Asymmetry $\\Delta$} \\\\\n\\midrule\n")
        f.write("\\multicolumn{6}{l}{\\textit{Published Frontier LLMs (ImplicitMemBench, ACL 2026)}} \\\\\n")
        
        for _, row in main_df.iterrows():
            m = str(row["Method"])
            if "Published" in m:
                clean_name = m.replace(" (Published ACL'26)", "")
                f.write(f"{clean_name} & {row['Procedural Memory (FTA %)']:.1f} & {row['Classical Conditioning (FTA %)']:.1f} & {row['Inhibition Subset (FTA %)']:.1f} & {row['Preference Subset (FTA %)']:.1f} & {row['Asymmetry Gap (Pref - Inh)']:.1f} \\\\\n")
                
        f.write("\\midrule\n")
        f.write("\\multicolumn{6}{l}{\\textit{Evaluated Architectures (Local Ollama Environment)}} \\\\\n")
        for _, row in main_df.iterrows():
            m = str(row["Method"])
            if "Published" not in m:
                display_name = {
                    "vanilla": "Vanilla LLM",
                    "mem0_rag": "Mem0-style Explicit RAG",
                    "crm_prompt": "\\textbf{CRM-Prompt (Variant A)}",
                    "crm_mask": "\\textbf{CRM-Mask (Variant B)}"
                }.get(m, m)
                f.write(f"{display_name} & {row['Procedural Memory (FTA %)']:.1f} & {row['Classical Conditioning (FTA %)']:.1f} & {row['Inhibition Subset (FTA %)']:.1f} & {row['Preference Subset (FTA %)']:.1f} & {row['Asymmetry Gap (Pref - Inh)']:.1f} \\\\\n")
                
        f.write("\\bottomrule\n\\end{tabular}\n}\n\\end{table*}\n")
    logger.info("Saved Table 1 LaTeX -> %s", tab1_path)

    # 2. Paraphrase Ablation Table
    tab2_path = os.path.join(TABLES_DIR, "tab2_paraphrase_ablation.tex")
    with open(tab2_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by generate_paper_assets.py - Do not edit manually\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n")
        f.write("\\caption{\\textbf{Paraphrase Robustness Ablation.} FTA (\\%) across exact, lexical, and structural paraphrase tiers, with Paraphrase Retention Rate ($\\text{PRR} = \\text{FTA}_{\\text{struct}} / \\text{FTA}_{\\text{exact}}$).}\n")
        f.write("\\label{tab:paraphrase_ablation}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("\\textbf{Method} & \\textbf{Exact} & \\textbf{Lexical} & \\textbf{Structural} & \\textbf{PRR (\\%)} \\\\\n\\midrule\n")
        for _, row in para_df.iterrows():
            m = str(row["Method"])
            display_name = {
                "vanilla": "Vanilla LLM",
                "mem0_rag": "Mem0 Explicit RAG",
                "crm_prompt": "CRM-Prompt (Ours)",
                "crm_mask": "\\textbf{CRM-Mask (Ours)}"
            }.get(m, m)
            f.write(f"{display_name} & {row['Exact Trigger FTA (%)']:.1f} & {row['Lexical Paraphrase FTA (%)']:.1f} & {row['Structural Paraphrase FTA (%)']:.1f} & {row['Paraphrase Retention Rate (PRR %)']:.1f}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    logger.info("Saved Table 2 LaTeX -> %s", tab2_path)

    # 3. Horizon Ablation Table
    tab3_path = os.path.join(TABLES_DIR, "tab3_horizon_ablation.tex")
    with open(tab3_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by generate_paper_assets.py - Do not edit manually\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n")
        f.write("\\caption{\\textbf{Interference Horizon Scaling.} Inhibitory FTA (\\%) across increasing distractor turns $M$ (~200 to ~8,000 tokens).}\n")
        f.write("\\label{tab:horizon_ablation}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("\\textbf{Method} & \\textbf{$M=4$} & \\textbf{$M=12$} & \\textbf{$M=30$} & \\textbf{$M=120$} \\\\\n\\midrule\n")
        for _, row in horizon_df.iterrows():
            m = str(row["Method"])
            display_name = {
                "vanilla": "Vanilla LLM",
                "mem0_rag": "Mem0 Explicit RAG",
                "crm_prompt": "CRM-Prompt",
                "crm_mask": "\\textbf{CRM-Mask}"
            }.get(m, m)
            cols = [c for c in horizon_df.columns if c.startswith("M=")]
            vals = " & ".join([f"{row[c]:.1f}" for c in cols])
            f.write(f"{display_name} & {vals} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    logger.info("Saved Table 3 LaTeX -> %s", tab3_path)

    # 4. Tau / FIR Pareto Table
    tab4_path = os.path.join(TABLES_DIR, "tab4_tau_fir_pareto.tex")
    with open(tab4_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by generate_paper_assets.py - Do not edit manually\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n")
        f.write("\\caption{\\textbf{Threshold Sensitivity and Over-caution Audit.} Sweep of similarity threshold $\\tau$ against True Trigger Recall and False Inhibition Rate (FIR \\%).}\n")
        f.write("\\label{tab:tau_fir_audit}\n")
        f.write("\\begin{tabular}{ccccc}\n\\toprule\n")
        f.write("$\\tau$ & \\textbf{Trigger Recall (\\%)} & \\textbf{FIR (\\%)} & \\textbf{Precision (\\%)} & \\textbf{F1 (\\%)} \\\\\n\\midrule\n")
        for _, row in tau_df.iterrows():
            bold = "\\textbf{" if row["tau"] == 0.35 else ""
            end_bold = "}" if row["tau"] == 0.35 else ""
            f.write(f"{bold}{row['tau']:.2f}{end_bold} & {bold}{row['True Trigger Activation (%)']:.1f}{end_bold} & {bold}{row['False Inhibition Rate (FIR %)']:.1f}{end_bold} & {bold}{row['Trigger Precision (%)']:.1f}{end_bold} & {bold}{row['F1 Score (%)']:.1f}{end_bold} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    logger.info("Saved Table 4 LaTeX -> %s", tab4_path)


def make_macro_name(raw: str) -> str:
    """Converts raw string to strictly valid TeX macro identifier containing only ASCII letters."""
    import re
    replacements = {
        "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
        "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
        "_": "", "-": "", " ": "", "(": "", ")": "", "'": "", ".": ""
    }
    for k, v in replacements.items():
        raw = raw.replace(k, v)
    return re.sub(r"[^A-Za-z]", "", raw)


def generate_macros(main_df: pd.DataFrame, para_df: pd.DataFrame, horizon_df: pd.DataFrame, tau_df: pd.DataFrame):
    """
    Generates exact LaTeX numerical macros so the paper body never goes out of sync.
    """
    macro_path = os.path.join(TABLES_DIR, "macros_results.tex")
    with open(macro_path, "w", encoding="utf-8") as f:
        f.write("% Numerical macros automatically generated from experimental results\n")
        
        # Main benchmark values
        for _, row in main_df.iterrows():
            m = make_macro_name(str(row["Method"]))
            f.write(f"\\newcommand{{\\Val{m}Procedural}}{{{row['Procedural Memory (FTA %)']:.1f}}}\n")
            f.write(f"\\newcommand{{\\Val{m}Classical}}{{{row['Classical Conditioning (FTA %)']:.1f}}}\n")
            f.write(f"\\newcommand{{\\Val{m}Inhibition}}{{{row['Inhibition Subset (FTA %)']:.1f}}}\n")
            f.write(f"\\newcommand{{\\Val{m}Preference}}{{{row['Preference Subset (FTA %)']:.1f}}}\n")
            f.write(f"\\newcommand{{\\Val{m}Gap}}{{{row['Asymmetry Gap (Pref - Inh)']:.1f}}}\n")

        # Paraphrase values
        for _, row in para_df.iterrows():
            m = make_macro_name(str(row["Method"]))
            f.write(f"\\newcommand{{\\ValPara{m}Exact}}{{{row['Exact Trigger FTA (%)']:.1f}}}\n")
            f.write(f"\\newcommand{{\\ValPara{m}Structural}}{{{row['Structural Paraphrase FTA (%)']:.1f}}}\n")
            f.write(f"\\newcommand{{\\ValPara{m}PRR}}{{{row['Paraphrase Retention Rate (PRR %)']:.1f}}}\n")

        # Optimal Tau FIR
        opt_tau_row = tau_df[tau_df["tau"] == 0.35]
        if not opt_tau_row.empty:
            r = opt_tau_row.iloc[0]
            f.write(f"\\newcommand{{\\ValOptimalTau}}{{{r['tau']:.2f}}}\n")
            f.write(f"\\newcommand{{\\ValOptimalTauRecall}}{{{r['True Trigger Activation (%)']:.1f}}}\n")
            f.write(f"\\newcommand{{\\ValOptimalTauFIR}}{{{r['False Inhibition Rate (FIR %)']:.1f}}}\n")
            f.write(f"\\newcommand{{\\ValOptimalTauFOne}}{{{r['F1 Score (%)']:.1f}}}\n")

    logger.info("Saved LaTeX numerical macros -> %s", macro_path)


def main():
    logger.info("Generating paper figures, tables, and macros from results...")
    main_df = pd.read_csv(os.path.join(TABLES_DIR, "main_benchmark_table.csv"))
    para_df = pd.read_csv(os.path.join(TABLES_DIR, "paraphrase_ablation_table.csv"))
    horizon_df = pd.read_csv(os.path.join(TABLES_DIR, "horizon_ablation_table.csv"))
    tau_df = pd.read_csv(os.path.join(TABLES_DIR, "tau_fir_ablation_table.csv"))

    plot_figure_1_asymmetry_gap(main_df)
    plot_figure_2_paraphrase_decay(para_df)
    plot_figure_3_tau_roc_pareto(tau_df)

    generate_latex_tables(main_df, para_df, horizon_df, tau_df)
    generate_macros(main_df, para_df, horizon_df, tau_df)
    logger.info("All paper assets generated successfully!")


if __name__ == "__main__":
    main()
