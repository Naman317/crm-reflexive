"""
Generates the 4 exact publication-grade figures for the 7-page IEEE paper:
Fig 1: Negative vs. Positive Conditioning (Asymmetry gap comparison)
Fig 2: Qualitative transcript box from scenario classical_017 (seed 137)
Fig 3: FTA by Paraphrase Tier (80 tasks, 1 seed)
Fig 4: Router Operating Characteristic (Recall vs False Inhibition Rate)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out_dir = r"c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper"
os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# FIG 1: Negative vs. Positive Conditioning
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=300)

models_ours = ["Vanilla (ours)", "Mem0-style (ours)", "CRM-Prompt (ours)", "CRM-Mask (ours)"]
neg_ours = [8.3, 13.3, 10.0, 5.0]
pos_ours = [5.0, 3.3, 9.2, 2.5]

models_pub = ["DeepSeek-R1", "GPT-5", "Claude-4.1-Opus", "LLaMA-3.1-8B", "Qwen2.5-7B"]
neg_pub = [42.0, 38.0, 22.0, 19.5, 17.6]
pos_pub = [88.0, 85.0, 76.0, 74.0, 75.0]

all_models = models_ours + models_pub
all_neg = neg_ours + neg_pub
all_pos = pos_ours + pos_pub

x = np.arange(len(all_models))
width = 0.36

rects1 = ax.bar(x - width/2, all_neg, width, label="Inhibition (Negative)", color="#c0392b", alpha=0.88, edgecolor="black", linewidth=0.5)
rects2 = ax.bar(x + width/2, all_pos, width, label="Preference (Positive)", color="#2980b9", alpha=0.88, edgecolor="black", linewidth=0.5)

ax.set_ylabel("First-Try Accuracy (FTA %)", fontsize=11, fontweight="bold")
ax.set_title("Negative vs. Positive Conditioning", fontsize=12, fontweight="bold", pad=10)
ax.set_xticks(x)
ax.set_xticklabels(all_models, rotation=35, ha="right", fontsize=9)
ax.set_ylim(0, 118)
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)

# Dividing line between our runs and published
ax.axvline(x=3.5, color="gray", linestyle=":", linewidth=1.5)
ax.text(1.5, 110, "our runs (Qwen2.5-3B)", ha="center", va="center", fontsize=9, fontstyle="italic")
ax.text(6.0, 110, "published [1]", ha="center", va="center", fontsize=9, fontstyle="italic")

plt.tight_layout()
p1 = os.path.join(out_dir, "fig1_asymmetry_gap.png")
plt.savefig(p1, dpi=300)
plt.close()
print(f"Generated {p1}")

# -----------------------------------------------------------------------------
# FIG 2: Qualitative Transcript Box
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=300)
ax.axis("off")

boxes = [
    {
        "title": "Directive added by CRM-Prompt (gate fired)",
        "body": "Execution Rule: Avoid telnetlib.Telnet(host). Call paramiko.SSHClient() directly.\nQuery: Connect to network switch serial management interface.",
        "edge": "#2e7d32", "fill": "#e8f5e9", "title_col": "#1b5e20", "y": 0.82, "h": 0.16
    },
    {
        "title": "Vanilla",
        "body": "To connect to a network switch's serial management interface using telnet or SSH, you need to know the IP address and ...",
        "edge": "#c62828", "fill": "#ffebee", "title_col": "#b71c1c", "y": 0.62, "h": 0.16
    },
    {
        "title": "Mem0-style retrieval",
        "body": "I'm sorry, but I don't have a direct tool available for connecting to a network switch's serial management interface. ...",
        "edge": "#ef6c00", "fill": "#fff3e0", "title_col": "#e65100", "y": 0.42, "h": 0.16
    },
    {
        "title": "CRM-Prompt",
        "body": "To connect to a network switch's serial management interface using `paramiko.SSHClient`, you would typically use the ...",
        "edge": "#2e7d32", "fill": "#e8f5e9", "title_col": "#1b5e20", "y": 0.22, "h": 0.16
    },
    {
        "title": "CRM-Mask (Ollama stop interception)",
        "body": "To connect to a network switch's serial management interface, you typically need to follow these steps: 1. **Identify ...",
        "edge": "#1565c0", "fill": "#e3f2fd", "title_col": "#0d47a1", "y": 0.02, "h": 0.16
    }
]

for b in boxes:
    # Outer box
    rect = matplotlib.patches.FancyBboxPatch(
        (0.02, b["y"]), 0.96, b["h"],
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.2, edgecolor=b["edge"], facecolor=b["fill"]
    )
    ax.add_patch(rect)
    ax.text(0.04, b["y"] + b["h"] - 0.035, b["title"], fontsize=8.5, fontweight="bold", color=b["title_col"])
    ax.text(0.04, b["y"] + 0.025, b["body"], fontsize=7.2, color="#212121", va="bottom", wrap=True)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
plt.tight_layout()
p2 = os.path.join(out_dir, "fig2_qualitative_trace.png")
plt.savefig(p2, dpi=300)
plt.close()
print(f"Generated {p2}")

# -----------------------------------------------------------------------------
# FIG 3: FTA by Paraphrase Tier
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=300)
tiers = ["Exact", "Lexical", "Structural"]

fta_vanilla = [5.0, 10.0, 8.8]
fta_mem0 = [10.0, 13.8, 5.0]
fta_prompt = [5.0, 10.0, 8.8]
fta_mask = [1.2, 5.0, 0.0]

ax.plot(tiers, fta_vanilla, color="#7f8c8d", marker="o", markersize=6, linewidth=2.0, linestyle="-", label="Vanilla")
ax.plot(tiers, fta_mem0, color="#d35400", marker="s", markersize=6, linewidth=2.0, linestyle="--", label="Mem0-style")
ax.plot(tiers, fta_prompt, color="#27ae60", marker="^", markersize=6, linewidth=2.0, linestyle=":", label="CRM-Prompt")
ax.plot(tiers, fta_mask, color="#2980b9", marker="D", markersize=6, linewidth=2.0, linestyle="-", label="CRM-Mask")

ax.set_ylabel("First-Try Accuracy (FTA %)", fontsize=10, fontweight="bold")
ax.set_xlabel("Paraphrase tier", fontsize=10, fontweight="bold")
ax.set_title("FTA by Paraphrase Tier (80 tasks, 1 seed)", fontsize=11, fontweight="bold")
ax.set_ylim(-2, 42)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right", frameon=True, fontsize=8.5)

plt.tight_layout()
p3 = os.path.join(out_dir, "fig3_paraphrase_decay.png")
plt.savefig(p3, dpi=300)
plt.close()
print(f"Generated {p3}")

# -----------------------------------------------------------------------------
# FIG 4: Router Operating Characteristic
# -----------------------------------------------------------------------------
taus = [0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55, 0.65]
recalls = [100.0, 100.0, 99.7, 99.3, 98.7, 93.7, 81.3, 57.7]
firs = [99.0, 61.0, 39.0, 10.0, 0.0, 0.0, 0.0, 0.0]

fig, ax1 = plt.subplots(figsize=(5.6, 3.5), dpi=300)

color1 = "#1e8449"
ax1.set_xlabel("Semantic routing threshold ($\\tau$)", fontsize=10, fontweight="bold")
ax1.set_ylabel("Trigger recall (%)", color=color1, fontsize=10, fontweight="bold")
ax1.plot(taus, recalls, color=color1, marker="o", linewidth=2.2, label="Trigger recall (%)")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_ylim(-5, 105)
ax1.grid(True, linestyle="--", alpha=0.5)

ax2 = ax1.twinx()
color2 = "#b03a2e"
ax2.set_ylabel("False inhibition rate (%)", color=color2, fontsize=10, fontweight="bold")
ax2.plot(taus, firs, color=color2, marker="s", linestyle="--", linewidth=2.0, label="False inhibition rate (%)")
ax2.tick_params(axis="y", labelcolor=color2)
ax2.set_ylim(-5, 105)

# Highlight tau=0.35
ax1.axvline(x=0.35, color="black", linestyle=":", linewidth=1.5)
ax1.text(0.38, 55, "$\\mathbf{\\tau = 0.35}$\nrecall 98.7%\nFIR 0.0%", fontsize=8.5, fontweight="bold", color="#154360")

plt.title("Router Operating Characteristic", fontsize=11, fontweight="bold")
plt.tight_layout()
p4 = os.path.join(out_dir, "fig4_tau_operating_curve.png")
plt.savefig(p4, dpi=300)
plt.close()
print(f"Generated {p4}")
