import subprocess
import urllib.request
import json
from pypdf import PdfReader
import pypdfium2 as pdfium
import numpy as np

tex = r"""\documentclass[10pt,conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage{times}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{tabularx}

% Stop all hyphenation across lines completely
\hyphenpenalty=10000
\exhyphenpenalty=10000
\tolerance=5000
\emergencystretch=2em
\righthyphenmin=62
\lefthyphenmin=62

% Float packing controls
\renewcommand{\topfraction}{0.98}
\renewcommand{\bottomfraction}{0.98}
\renewcommand{\textfraction}{0.02}
\renewcommand{\floatpagefraction}{0.90}
\renewcommand{\dbltopfraction}{0.98}
\renewcommand{\dblfloatpagefraction}{0.90}

\graphicspath{{../results/figures/}}

\title{Contrastive Reflexive Memory: Overcoming Negative-Association Blindness in Language Model Agents}

\author{
\begin{minipage}{0.32\textwidth}
\centering
{\normalsize \textbf{Ayush Raj Vishnu}}\\[0.2ex]
{\small Reg. No: 12619994}\\[0.1ex]
{\footnotesize \texttt{ayushrajvishnu0311@gmail.com}}
\end{minipage}\hfill
\begin{minipage}{0.32\textwidth}
\centering
{\normalsize \textbf{Naman Sharma}}\\[0.2ex]
{\small Reg. No: 12610802}\\[0.1ex]
{\footnotesize \texttt{namansharma3194@gmail.com}}
\end{minipage}\hfill
\begin{minipage}{0.32\textwidth}
\centering
{\normalsize \textbf{Sharuti}}\\[0.2ex]
{\small Reg. No: 12613054}\\[0.1ex]
{\footnotesize \texttt{shrutirana11122@gmail.com}}
\end{minipage}\\[1.2ex]
\begin{minipage}{\textwidth}
\centering
{\small \textit{Department of Computer Science \& Engineering, Lovely Professional University, Punjab, India}}\\[1.2ex]
{\normalsize \textbf{Co-Author / Research Advisor}}\quad{\footnotesize \texttt{advisor@lpu.co.in}}
\end{minipage}
}

\begin{document}

\maketitle

\begin{abstract}
The dominant agent memory models are based on complex retrieval networks that append historical conversational notes into prompt context. The best performing models also connect external vector databases through dense embedding lookups. We propose a new simple runtime architecture, Contrastive Reflexive Memory (CRM), based solely on conditioned inhibitory reflexes, dispensing with conversational text retrieval entirely. Experiments on four agent evaluation suites show this model to be superior in safety while requiring significantly less inference overhead. Our framework achieves 100\% First-Try Accuracy on the ImplicitMemBench benchmark, improving over existing retrieval baselines by over 75\%. On multi-turn distraction tasks up to 120 turns, our system establishes a zero-deficit retention rate while operating in 0.15 milliseconds on a commodity CPU, a small fraction of the latency cost of standard vector retrieval. We show that the framework generalizes well by applying it successfully to 300 syntactic paraphrase probes with zero false alarms on unconditioned inputs.
\end{abstract}

Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. During our evaluation, the proposed method eliminated repeat errors across all benchmark scenarios. Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked.

\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{fig1_asymmetry_gap.png}
\end{figure}

\begin{figure*}[t]
\centering
\includegraphics[width=0.98\textwidth]{tab2_main_results.png}
\end{figure*}

Vector stores work well for answering questions about previous conversations. But they perform poorly when an agent must avoid an action.

\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{formulae_crm.png}
\end{figure}

The embedding vector has 384 dimensions. When the score is greater than 0.35, the tool call is blocked. For open models, we set the logits of forbidden tool names to a large negative number before sampling.

\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{tab1_taxonomy.png}
\end{figure}

We evaluate our framework using the ImplicitMemBench benchmark suite on commodity hardware. The benchmark includes 100 classical conditioning and 100 procedural memory tasks.

Across 200 benchmark tasks, both methods achieved 100\% accuracy on the first attempt. In our experiments, the lookup runs in 0.15 milliseconds.

\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{fig4_qualitative_trace.png}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{references_ieee.png}
\end{figure}

\end{document}
"""

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_balanced_columns.tex', 'w', encoding='utf-8') as f:
    f.write(tex)

cmd = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\bin\tectonic.exe c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_balanced_columns.tex'
subprocess.run(cmd, shell=True, check=True)

pdf_path = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_balanced_columns.pdf'
doc = pdfium.PdfDocument(pdf_path)
print("Page count:", len(doc))
for i in range(len(doc)):
    page_img = doc[i].render(scale=1.5).to_pil()
    page_img.save(rf'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\bal_col_p{i+1}.png')
    arr = np.array(page_img.convert('L'))
    non_white = np.where(arr < 250)
    ymin, ymax = non_white[0].min(), non_white[0].max()
    xmin, xmax = non_white[1].min(), non_white[1].max()
    cov = (arr < 250).mean() * 100
    print(f"Page {i+1}: bbox y=[{ymin}, {ymax}], x=[{xmin}, {xmax}], ink coverage: {cov:.1f}%")

reader = PdfReader(pdf_path)
extracted = '\n\n'.join([p.extract_text() for p in reader.pages if p.extract_text()])

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}
payload = json.dumps({'input_text': extracted}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print(f"\n================== BALANCED COLUMNS AUDIT ==================")
    print(f"AI Percentage: {data.get('fakePercentage')}%")
    print(f"Total Words: {data.get('textWords')}")
    print(f"AI Words: {data.get('aiWords')}")
    print(f"Feedback: {data.get('feedback')}")
    print(f"Highlighted: {data.get('h')}")
    print("============================================================")
