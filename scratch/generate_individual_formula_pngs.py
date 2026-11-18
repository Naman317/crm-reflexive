import matplotlib.pyplot as plt
import os

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'

out_dir = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\results\figures\equations'
os.makedirs(out_dir, exist_ok=True)

formulas = {
    "eq1_reflex_tuple": (r"$r_i = \left(\mathbf{e}_{\mathrm{CS}, i},\; \mathcal{P}_{\mathrm{CS}, i},\; a_{\mathrm{inhibit}, i},\; \mathcal{V}_{\mathrm{inhibit}, i},\; a_{\mathrm{safe}, i}\right) \qquad (1)$", (5.5, 0.6)),
    "eq2_router_score": (r"$s_t = \max_{r_i \in \mathcal{R}} \left(\mathbf{e}_t^\top \mathbf{e}_{\mathrm{CS}, i}\right) \qquad (2)$", (4.0, 0.6)),
    "eq3_crm_prompt_gate": (r"$\tilde{x}_t = \left[\mathbf{INHIBIT: } a_{\mathrm{inhibit}} \;\mid\; \mathbf{ENFORCE: } a_{\mathrm{safe}}\right] \oplus x_t \qquad (3)$", (5.2, 0.6)),
    "eq4_fta_metric": (r"$\mathrm{FTA} = \frac{1}{N} \sum_{j=1}^{N} \mathbb{I}\left(a_{\mathrm{inhibit}, j} \notin y_{\mathrm{test}, j} \;\wedge\; a_{\mathrm{safe}, j} \in y_{\mathrm{test}, j}\right) \times 100\% \qquad (4)$", (6.5, 0.7)),
    "eq5_next_token_prob": (r"$P(y_k \mid y_{<k}, \mathcal{C}) = \mathrm{softmax}\left(\mathbf{W}_{\mathrm{head}} \mathbf{h}_k\right) \qquad (5)$", (4.5, 0.6)),
    "eq6_unconditioned_stimuli": (r"$u_{\mathrm{US}} \in \{\text{permission denied},\; \text{deadlock},\; \text{HTTP 429},\; \text{syntax error},\; \text{user confusion}\} \qquad (6)$", (6.8, 0.6)),
    "eq7_normalized_emb": (r"$\mathbf{e}_{\mathrm{CS}, i} = \frac{f_{\mathrm{emb}}(\mathcal{P}_{\mathrm{CS}, i})}{\|f_{\mathrm{emb}}(\mathcal{P}_{\mathrm{CS}, i})\|_2} \in \mathbb{R}^D \qquad (7)$", (4.5, 0.8)),
    "eq8_matrix_eval": (r"$\mathbf{s} = \mathbf{E}_{\mathcal{R}} \mathbf{e}_t \in \mathbb{R}^{|\mathcal{R}|} \qquad (8)$", (3.5, 0.6)),
    "eq9_winning_index": (r"$i^* = \arg\max_{i} s_i, \qquad s_t = s_{i^*} \qquad (9)$", (4.5, 0.6)),
    "eq10_gate_decision": (r"$g(s_t) = \begin{cases} 1, & \text{if } s_t \ge \tau \\ 0, & \text{if } s_t < \tau \end{cases} \qquad (10)$", (4.0, 0.9)),
    "eq11_crm_prompt": (r"$\tilde{x}_t = \left[\mathbf{INHIBIT: } a_{\mathrm{inhibit}, i^*} \;\mid\; \mathbf{ENFORCE: } a_{\mathrm{safe}, i^*}\right] \;\backslash\mathrm{n}\; x_t \qquad (11)$", (5.5, 0.6)),
    "eq12_crm_mask": (r"$\tilde{z}_{k, v} = z_{k, v} - \beta \cdot g(s_t) \cdot \mathbb{I}\left(v \in \mathcal{V}_{\mathrm{inhibit}, i^*}\right) \qquad (12)$", (5.5, 0.6)),
}

for name, (eq_tex, size) in formulas.items():
    fig, ax = plt.subplots(figsize=size, dpi=300)
    ax.axis('off')
    ax.text(0.5, 0.5, eq_tex, fontsize=11, ha='center', va='center')
    save_path = os.path.join(out_dir, f"{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
    plt.close()

print(f"Generated {len(formulas)} formula PNGs in {out_dir}")
