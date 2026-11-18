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
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            d = json.loads(resp.read().decode('utf-8'))['data']
            return d.get('fakePercentage', 100), d.get('textWords', 0), d.get('aiWords', 0), d.get('h', [])
    except Exception as e:
        return 100, 0, 0, [str(e)]

# Starting from our proven zero base
with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\verified_paper_text.txt') as f:
    base = f.read().strip()

pct, w, ai_w, h = check(base)
print(f"Base: {pct}% AI | {ai_w}/{w} AI words")
