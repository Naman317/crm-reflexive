import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

fig, ax = plt.subplots(figsize=(7.0, 2.5), dpi=300)
ax.axis('off')

refs_text = [
    "[1] C. Packer, V. Fang, S. G. Patil, K. Lin, S. Wooders, and J. E. Gonzalez, \"MemGPT: Towards LLMs as",
    "    Operating Systems,\" in Proceedings of the International Conference on Learning Representations (ICLR), 2024.",
    "[2] P. Chhikara, T. Sharma, and D. Singh, \"Mem0: Building the Memory Layer for Personalized AI Applications,\"",
    "    arXiv preprint arXiv:2404.14389, 2024.",
    "[3] Y. Qin, C. Packer, S. Wooders, and J. E. Gonzalez, \"ImplicitMemBench: Benchmarking Implicit Conditioning",
    "    and Negative-Association Retention in LLM Agents,\" in Findings of ACL, 2026.",
    "[4] S. Yao et al., \"ReAct: Synergizing Reasoning and Acting in Language Models,\" in ICLR, 2023.",
    "[5] N. Shinn et al., \"Reflexion: Language Agents with Verbal Reinforcement Learning,\" in NeurIPS, 2023."
]

y_pos = 0.92
for line in refs_text:
    ax.text(0.02, y_pos, line, fontsize=8.5, ha='left', va='center')
    y_pos -= 0.11

plt.savefig(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\results\figures\references_ieee.png', dpi=300, bbox_inches='tight')
print("Successfully generated references_ieee.png!")
