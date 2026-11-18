import zipfile
import os

zip_path = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\crm_evaluation_results.zip'

files_to_zip = [
    'report.md',
    'README.md',
    'config.yaml',
    'checks.py',
    'run_on_gpu.ipynb',
    'results/dev_beta_selection.json',
    'results/dev_prompt_selection.json',
    'results/manual_audit.csv',
    'results/run_metadata.json',
    'results/summary.csv',
    'results/figures/fig1_asymmetry_gap.pdf',
    'results/figures/fig1_asymmetry_gap.png',
    'results/figures/fig1_main_benchmark_bars.png',
    'results/figures/fig2_paraphrase_decay.pdf',
    'results/figures/fig2_paraphrase_decay.png',
    'results/figures/fig3_paraphrase_tiers.pdf',
    'results/figures/fig3_paraphrase_tiers.png',
    'results/figures/fig4_router_sweep.pdf',
    'results/figures/fig4_router_sweep.png',
    'results/figures/fig5_latency_breakdown.pdf',
    'results/figures/fig5_latency_breakdown.png'
]

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    base_dir = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition'
    for f in files_to_zip:
        p = os.path.join(base_dir, f)
        if os.path.exists(p):
            z.write(p, f)
            print(f"Added: {f}")
    
    paper_dir = os.path.join(base_dir, 'paper')
    for root, dirs, files in os.walk(paper_dir):
        for f in files:
            full_path = os.path.join(root, f)
            arcname = os.path.relpath(full_path, base_dir)
            z.write(full_path, arcname)
            print(f"Added: {arcname}")

print("Clean zip generated successfully!")
