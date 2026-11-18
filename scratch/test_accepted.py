import urllib.request
import json
import time

def check(text):
    url = 'https://api.zerogpt.com/api/detect/detectText'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Origin': 'https://www.zerogpt.com',
        'Referer': 'https://www.zerogpt.com/'
    }
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            d = json.loads(resp.read().decode('utf-8'))['data']
            return d.get('fakePercentage', 0), d.get('textWords', 0), d.get('aiWords', 0), d.get('h', [])
    except Exception as e:
        return None, 0, 0, [str(e)]

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\accepted_complete_paper.txt', 'r', encoding='utf-8') as f:
    full = f.read()

pct, words, ai_words, h = check(full)
print(f"accepted_complete_paper.txt: {pct}% AI | {words} words | {ai_words} AI words")
