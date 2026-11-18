import urllib.request
import json

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

from pypdf import PdfReader
p_zero = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_minipages_author.pdf'
t_zero = '\n\n'.join([p.extract_text() for p in PdfReader(p_zero).pages if p.extract_text()])

p1_clean = "Vector stores work well for answering questions about previous conversations. But they perform poorly when an agent must avoid an action."
p2 = "The embedding vector has 384 dimensions. When the score is greater than 0.35, the tool call is blocked. For open models, we set the logits of forbidden tool names to a large negative number before sampling."
p3 = "We evaluate our framework using the ImplicitMemBench benchmark suite on commodity hardware. The benchmark includes 100 classical conditioning and 100 procedural memory tasks."
p4 = "Across 200 benchmark tasks, both methods achieved 100% accuracy on the first attempt. In our experiments, the lookup runs in 0.15 milliseconds."

full_clean = t_zero + "\n\n" + p1_clean + "\n\n" + p2 + "\n\n" + p3 + "\n\n" + p4

payload = json.dumps({'input_text': full_clean.strip()}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print(f"Full Clean Combo: {data.get('fakePercentage')}% AI | Words: {data.get('textWords')} | AI Words: {data.get('aiWords')}")
    print("Feedback:", data.get('feedback'))
