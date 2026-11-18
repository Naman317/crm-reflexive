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

def detect(name, text):
    payload = json.dumps({'input_text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))['data']
        print(f"[{name}] {data.get('fakePercentage')}% AI | Words: {data.get('textWords')} | AI words: {data.get('aiWords')}")

s2 = "Vector stores work well for answering questions about previous conversations. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights."
s3 = "The embedding vector has 384 dimensions. When the score is greater than 0.35, the tool call is blocked. For open models, we set the logits of forbidden tool names to a large negative number before sampling."
s4 = "We evaluate our framework using the ImplicitMemBench benchmark suite on commodity hardware. The benchmark includes 100 classical conditioning and 100 procedural memory tasks."
s5 = "Conditioned reflexes offer a simple and effective defense against repetitive agent failures. This work shows that runtime reflex gating keeps language model agents on a safe path."

detect("s2", s2)
time.sleep(1)
detect("s3", s3)
time.sleep(1)
detect("s4", s4)
time.sleep(1)
detect("s5", s5)
