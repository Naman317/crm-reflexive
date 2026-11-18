import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(6.5, 2.0), dpi=300)
ax.axis('off')

# Equations written with standard matplotlib mathtext
eq1 = r"$\mathbf{r}_i = \left(\mathbf{e}_{\mathrm{CS}, i},\; \mathcal{P}_{\mathrm{CS}, i},\; a_{\mathrm{inhibit}, i},\; \mathcal{V}_{\mathrm{inhibit}, i},\; a_{\mathrm{safe}, i}\right) \quad \text{--- (1)}$"
eq2 = r"$s_t = \max_{\mathbf{r}_i \in \mathcal{R}} \left(\mathbf{e}_t^\top \mathbf{e}_{\mathrm{CS}, i}\right) \geq 0.35 \quad \text{--- (2)}$"
eq3 = r"$\tilde{x}_t = \left[\mathbf{INHIBIT: } a_{\mathrm{inhibit}} \;\mid\; \mathbf{ENFORCE: } a_{\mathrm{safe}}\right] \oplus x_t \quad \text{--- (3)}$"

ax.text(0.5, 0.80, eq1, fontsize=11, ha='center', va='center')
ax.text(0.5, 0.48, eq2, fontsize=11, ha='center', va='center')
ax.text(0.5, 0.16, eq3, fontsize=11, ha='center', va='center')

plt.savefig(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\results\figures\formulae_crm.png', dpi=300, bbox_inches='tight')
print("Successfully generated formulae_crm.png!")
