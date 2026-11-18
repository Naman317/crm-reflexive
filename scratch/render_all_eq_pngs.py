import subprocess
import os
import pypdfium2 as pdfium

out_dir = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\results\figures\equations'
os.makedirs(out_dir, exist_ok=True)
scratch_dir = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch'

formulas = {
    "eq1_reflex_tuple": r"r_i = \left(\mathbf{e}_{\mathrm{CS}, i},\; \mathcal{P}_{\mathrm{CS}, i},\; a_{\mathrm{inhibit}, i},\; \mathcal{V}_{\mathrm{inhibit}, i},\; a_{\mathrm{safe}, i}\right) \qquad (1)",
    "eq2_router_score": r"s_t = \max_{r_i \in \mathcal{R}} \left(\mathbf{e}_t^\top \mathbf{e}_{\mathrm{CS}, i}\right) \qquad (2)",
    "eq3_crm_prompt_gate": r"\tilde{x}_t = \left[\mathbf{INHIBIT: } a_{\mathrm{inhibit}} \;\mid\; \mathbf{ENFORCE: } a_{\mathrm{safe}}\right] \oplus x_t \qquad (3)",
    "eq4_fta_metric": r"\mathrm{FTA} = \frac{1}{N} \sum_{j=1}^{N} \mathbb{I}\left(a_{\mathrm{inhibit}, j} \notin y_{\mathrm{test}, j} \;\wedge\; a_{\mathrm{safe}, j} \in y_{\mathrm{test}, j}\right) \times 100\% \qquad (4)",
    "eq5_next_token_prob": r"P(y_k \mid y_{<k}, \mathcal{C}) = \mathrm{softmax}\left(\mathbf{W}_{\mathrm{head}} \mathbf{h}_k\right) \qquad (5)",
    "eq6_unconditioned_stimuli": r"u_{\mathrm{US}} \in \{\text{permission denied},\; \text{deadlock},\; \text{HTTP 429},\; \text{syntax error},\; \text{user confusion}\} \qquad (6)",
    "eq7_normalized_emb": r"\mathbf{e}_{\mathrm{CS}, i} = \frac{f_{\mathrm{emb}}(\mathcal{P}_{\mathrm{CS}, i})}{\|f_{\mathrm{emb}}(\mathcal{P}_{\mathrm{CS}, i})\|_2} \in \mathbb{R}^D \qquad (7)",
    "eq8_matrix_eval": r"\mathbf{s} = \mathbf{E}_{\mathcal{R}} \mathbf{e}_t \in \mathbb{R}^{|\mathcal{R}|} \qquad (8)",
    "eq9_winning_index": r"i^* = \arg\max_{i} s_i, \qquad s_t = s_{i^*} \qquad (9)",
    "eq10_gate_decision": r"g(s_t) = \begin{cases} 1, & \text{if } s_t \ge \tau \\ 0, & \text{if } s_t < \tau \end{cases} \qquad (10)",
    "eq11_crm_prompt": r"\tilde{x}_t = \left[\mathbf{INHIBIT: } a_{\mathrm{inhibit}, i^*} \;\mid\; \mathbf{ENFORCE: } a_{\mathrm{safe}, i^*}\right] \;\backslash\mathrm{n}\; x_t \qquad (11)",
    "eq12_crm_mask": r"\tilde{z}_{k, v} = z_{k, v} - \beta \cdot g(s_t) \cdot \mathbb{I}\left(v \in \mathcal{V}_{\mathrm{inhibit}, i^*}\right) \qquad (12)",
}

for name, eq_code in formulas.items():
    tex_path = os.path.join(scratch_dir, f"{name}.tex")
    pdf_path = os.path.join(scratch_dir, f"{name}.pdf")
    png_path = os.path.join(out_dir, f"{name}.png")
    
    tex_content = rf"""\documentclass[preview,border=4pt]{{standalone}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{times}}
\begin{{document}}
\[
{eq_code}
\]
\end{{document}}"""
    
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
        
    cmd = rf'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\bin\tectonic.exe {tex_path}'
    subprocess.run(cmd, shell=True, capture_output=True)
    
    if os.path.exists(pdf_path):
        doc = pdfium.PdfDocument(pdf_path)
        img = doc[0].render(scale=300/72).to_pil()
        img.save(png_path)
        print(f"Generated {name}.png")
    else:
        print(f"FAILED {name}")

print(f"Done rendering all equation PNGs in {out_dir}")
