import subprocess

test_tex = r"""\documentclass[10pt,conference]{IEEEtran}
\usepackage{times}
\begin{document}
We propose a new architecture, Contrastive Reflexive Memory (CRM), based solely on conditioned inhibitory reflexes, dispensing with conversational text retrieval entirely. Experiments on four agent evaluation suites show this model to be superior in safety.
\end{document}"""

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\test_font.tex', 'w') as f:
    f.write(test_tex)

res = subprocess.run([r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\bin\tectonic.exe', r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\test_font.tex'], capture_output=True, text=True)

import pypdf
reader = pypdf.PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\test_font.pdf')
txt = reader.pages[0].extract_text()
print('Extracted text:', repr(txt))
print('Has "entirely .":', 'entirely .' in txt)
print('Has "First-T ry":', 'First-T ry' in txt)
print('Has "W e":', 'W e' in txt)
