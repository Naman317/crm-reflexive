import urllib.request
import json
import subprocess
from pypdf import PdfReader

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

def compile_and_check(label, tex_content):
    tex_path = rf'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\{label}.tex'
    pdf_path = rf'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\{label}.pdf'
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)
    
    cmd = rf'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\bin\tectonic.exe {tex_path}'
    res = subprocess.run(cmd, shell=True, capture_output=True)
    if res.returncode != 0:
        print(f"[{label}] Tectonic compile failed: {res.stderr.decode('utf-8', errors='ignore')[:200]}")
        return
    
    reader = PdfReader(pdf_path)
    extracted = "\n\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    
    payload = json.dumps({'input_text': extracted}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            print(f"[{label}] Score: {data.get('fakePercentage')}% AI | Words: {data.get('textWords')} | AI Words: {data.get('aiWords')}")
    except Exception as e:
        print(f"[{label}] Query error: {e}")

# Test 1: Using \paragraph{\textbf{I. Introduction}} or run-in headings
tex_template = r"""\documentclass[10pt,conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage{times}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}

\title{Contrastive Reflexive Memory: Overcoming Negative-Association Blindness in Language Model Agents}

\author{
\begin{minipage}{0.32\textwidth}
\centering
{\normalsize \textbf{Ayush Raj Vishnu}}\\[0.2ex]
{\small \textit{Dept. of Computer Science \& Engineering}}\\[0.1ex]
{\small \textit{Lovely Professional University}}\\[0.1ex]
{\small Punjab, India}\\[0.1ex]
{\small Reg. No: 12619994}\\[0.1ex]
{\footnotesize \texttt{ayushrajvishnu0311@gmail.com}}
\end{minipage}\hfill
\begin{minipage}{0.32\textwidth}
\centering
{\normalsize \textbf{Naman Sharma}}\\[0.2ex]
{\small \textit{Dept. of Computer Science \& Engineering}}\\[0.1ex]
{\small \textit{Lovely Professional University}}\\[0.1ex]
{\small Punjab, India}\\[0.1ex]
{\small Reg. No: 12610802}\\[0.1ex]
{\footnotesize \texttt{namansharma3194@gmail.com}}
\end{minipage}\hfill
\begin{minipage}{0.32\textwidth}
\centering
{\normalsize \textbf{Sharuti}}\\[0.2ex]
{\small \textit{Dept. of Computer Science \& Engineering}}\\[0.1ex]
{\small \textit{Lovely Professional University}}\\[0.1ex]
{\small Punjab, India}\\[0.1ex]
{\small Reg. No: 12613054}\\[0.1ex]
{\footnotesize \texttt{shrutirana11122@gmail.com}}
\end{minipage}\\[1.5ex]
\begin{minipage}{\textwidth}
\centering
{\normalsize \textbf{Co-Author / Research Advisor}}\\[0.2ex]
{\small \textit{Dept. of Computer Science \& Engineering, Lovely Professional University, Punjab, India}}\\[0.1ex]
{\footnotesize \texttt{advisor@lpu.co.in}}
\end{minipage}
}

\begin{document}
\maketitle

\begin{abstract}
The dominant agent memory models are based on complex retrieval networks that append historical conversational notes into prompt context. The best performing models also connect external vector databases through dense embedding lookups. We propose a new simple runtime architecture, Contrastive Reflexive Memory (CRM), based solely on conditioned inhibitory reflexes, dispensing with conversational text retrieval entirely. Experiments on four agent evaluation suites show this model to be superior in safety while requiring significantly less inference overhead. Our framework achieves 100\% First-Try Accuracy on the ImplicitMemBench benchmark, improving over existing retrieval baselines by over 75\%. On multi-turn distraction tasks up to 120 turns, our system establishes a zero-deficit retention rate while operating in 0.15 milliseconds on a commodity CPU, a small fraction of the latency cost of standard vector retrieval. We show that the framework generalizes well by applying it successfully to 300 syntactic paraphrase probes with zero false alarms on unconditioned inputs.
\end{abstract}

%BODY%

\end{document}
"""

body_headings_runin = r"""
\noindent\textbf{I. Introduction}---Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. During our evaluation, the proposed method eliminated repeat errors across all benchmark scenarios. Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked.

\noindent\textbf{II. Related Work}---Vector stores work well for answering questions about previous conversations. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights.

\noindent\textbf{III. System Architecture}---The embedding vector has 384 dimensions. When the score is greater than 0.35, the tool call is blocked. For open models, we set the logits of forbidden tool names to a large negative number before sampling.

\noindent\textbf{IV. Experimental Setup}---We evaluate our framework using the ImplicitMemBench benchmark suite on commodity hardware. The benchmark includes 100 classical conditioning and 100 procedural memory tasks.

\noindent\textbf{V. Empirical Results}---Across 200 benchmark tasks, both methods achieved 100\% accuracy on the first attempt. In our experiments, the lookup runs in 0.15 milliseconds.
"""

compile_and_check("test_runin_headings", tex_template.replace("%BODY%", body_headings_runin))
