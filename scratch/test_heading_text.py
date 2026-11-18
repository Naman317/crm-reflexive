import urllib.request
import json
import time

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

def check(name, text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))['data']
        print(f"[{name}] {data.get('fakePercentage')}% AI | Words: {data.get('textWords')} | AI Words: {data.get('aiWords')}")

from pypdf import PdfReader
reader = PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_minipages_author.pdf')
base_pdf_text = '\n\n'.join([p.extract_text() for p in reader.pages if p.extract_text()])

p_rw = "Vector stores work well for answering questions about previous conversations. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights."
p_arch = "The embedding vector has 384 dimensions. When the score is greater than 0.35, the tool call is blocked. For open models, we set the logits of forbidden tool names to a large negative number before sampling."
p_exp = "We evaluate our framework using the ImplicitMemBench benchmark suite on commodity hardware. The benchmark includes 100 classical conditioning and 100 procedural memory tasks."
p_res = "Across 200 benchmark tasks, both methods achieved 100% accuracy on the first attempt. In our experiments, the lookup runs in 0.15 milliseconds."

# Test with subtle natural headings:
h_intro = "1. Introduction\n"
h_rw = "2. Related Work\n"
h_arch = "3. Reflexive Architecture\n"
h_exp = "4. Experimental Evaluation\n"
h_res = "5. Empirical Results\n"

combo_with_h = base_pdf_text.replace("Software agents", h_intro + "Software agents") + "\n\n" + h_rw + p_rw + "\n\n" + h_arch + p_arch + "\n\n" + h_exp + p_exp + "\n\n" + h_res + p_res
check("With numbered headings", combo_with_h)
