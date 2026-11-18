import urllib.request
import json
import time

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\verified_paper_text.txt', 'r', encoding='utf-8') as f:
    text = f.read().strip()

paras = text.split('\n\n')

def test_zerogpt(t):
    url = 'https://api.zerogpt.com/api/detect/detectText'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Origin': 'https://www.zerogpt.com',
        'Referer': 'https://www.zerogpt.com/'
    }
    payload = json.dumps({'input_text': t.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            return data.get('fakePercentage'), data.get('feedback'), data.get('aiWords')
    except Exception as e:
        return None, str(e), None

print(f"Total paragraphs in verified_paper_text: {len(paras)}")
for i, p in enumerate(paras):
    pct, fb, ai_w = test_zerogpt(p)
    print(f"Para {i+1} (length {len(p.split())} words) -> AI%: {pct}%, Feedback: {fb}, AI words: {ai_w}")
    time.sleep(1)

# Also test Para 1 + Para 2
pct, fb, ai_w = test_zerogpt(paras[0] + "\n\n" + paras[1])
print(f"Para 1+2 -> AI%: {pct}%, Feedback: {fb}")

# Also test Para 3 (Abstract) + Para 4 (Body)
pct, fb, ai_w = test_zerogpt(paras[2] + "\n\n" + paras[3])
print(f"Para 3+4 -> AI%: {pct}%, Feedback: {fb}")
