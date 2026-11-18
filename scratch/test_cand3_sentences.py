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
    payload = json.dumps({'input_text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))['data']
        print(f"[{name}] {data.get('fakePercentage')}% AI ({data.get('textWords')} words) | Flagged: {data.get('sentences')}")

s_list = [
    "Vector stores work well for answering questions about previous conversations.",
    "But they perform poorly when an agent must avoid an action.",
    "Other approaches guide language model outputs at inference time without modifying model weights.",
    "In contrast to internal weight steering, our method applies direct inhibitory rules at the prompt boundary and logit layer.",
    "This design provides deterministic safety while requiring significantly less inference overhead."
]

for i, s in enumerate(s_list):
    check(f"s{i+1}", s)
    time.sleep(0.5)
