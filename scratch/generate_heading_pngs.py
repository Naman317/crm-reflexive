import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']

headings = [
    ("sec1_intro.png", "I. INTRODUCTION"),
    ("sec2_related_work.png", "II. RELATED WORK"),
    ("sec3_architecture.png", "III. SYSTEM ARCHITECTURE"),
    ("sec4_experimental_setup.png", "IV. EXPERIMENTAL SETUP"),
    ("sec5_results.png", "V. EMPIRICAL RESULTS"),
    ("sec6_conclusion.png", "VI. CONCLUSION")
]

for fname, text in headings:
    fig, ax = plt.subplots(figsize=(3.4, 0.35), dpi=300)
    ax.axis('off')
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=9.5, fontweight='bold', fontfamily='serif')
    plt.tight_layout(pad=0)
    plt.savefig(rf'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\results\figures\{fname}', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()

print("Generated all 6 heading PNGs at 300 DPI.")
