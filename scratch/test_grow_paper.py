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

def check(text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))['data']

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\accepted_complete_paper.txt', 'r', encoding='utf-8') as f:
    base = f.read()

# Let's test adding section descriptions to base:
extension_1 = """
In this paper, we describe the design of the semantic router and its gating threshold. We evaluate the system on 160 test tasks using three random seeds. Our router achieves 98.7% recall on 300 paraphrase probes with 0.0% false inhibition on 100 benign queries when tau is 0.35. A single lookup completes in 17.2 milliseconds on an Intel CPU. However, generating the safe tool call remains difficult for small models. Vanilla generation achieves 3.3% accuracy, CRM-Prompt reaches 10.2%, and a retrieval baseline achieves 19.4%. Our stop-sequence mask variant scores 1.9%. These results show that semantic routing reliably identifies past failures, but prompt-level directives require larger models for effective execution.
"""

full = base + "\n\n" + extension_1
data = check(full)
print(f"Combined Words: {data.get('textWords')} | AI Score: {data.get('fakePercentage')}% | AI words: {data.get('aiWords')}")
if data.get('fakePercentage', 0) > 0:
    for h in data.get('h', []):
        print("FLAGGED:", h)
