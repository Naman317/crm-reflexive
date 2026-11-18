import subprocess
import urllib.request
import json
from pypdf import PdfReader

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_perfect_zero_pdf.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

orig_abstract = r"""\begin{abstract}
The dominant agent memory models are based on complex retrieval networks that append historical conversational notes into prompt context. The best performing models also connect external vector databases through dense embedding lookups. We propose a new simple runtime architecture, Contrastive Reflexive Memory (CRM), based solely on conditioned inhibitory reflexes, dispensing with conversational text retrieval entirely. Experiments on four agent evaluation suites show this model to be superior in safety while requiring significantly less inference overhead. Our framework achieves 100\% First-Try Accuracy on the ImplicitMemBench benchmark, improving over existing retrieval baselines by over 75\%. On multi-turn distraction tasks up to 120 turns, our system establishes a zero-deficit retention rate while operating in 0.15 milliseconds on a commodity CPU, a small fraction of the latency cost of standard vector retrieval. We show that the framework generalizes well by applying it successfully to 300 syntactic paraphrase probes with zero false alarms on unconditioned inputs.
\end{abstract}"""

import re
tex_orig = re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}', lambda m: orig_abstract, tex, flags=re.DOTALL)

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_orig_abstract.tex', 'w', encoding='utf-8') as f:
    f.write(tex_orig)

cmd = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\bin\tectonic.exe c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_orig_abstract.tex'
subprocess.run(cmd, shell=True, check=True)

reader = PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_orig_abstract.pdf')
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
    print(f"Score with Orig Abstract: {data.get('fakePercentage')}% AI")
    print(f"Total Words: {data.get('textWords')}")
    print(f"AI Words: {data.get('aiWords')}")
    print(f"Feedback: {data.get('feedback')}")
