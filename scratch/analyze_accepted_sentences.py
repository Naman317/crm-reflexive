import urllib.request
import json

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\accepted_complete_paper.txt', 'r', encoding='utf-8') as f:
    text = f.read()

payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req, timeout=10) as resp:
    res = json.loads(resp.read().decode('utf-8'))
    data = res.get('data', {})
    print("Total Words:", data.get('textWords'))
    print("AI Score:", data.get('fakePercentage'), "%")
    print("AI Words:", data.get('aiWords'))
    print("\nFlagged sentences:")
    for s in data.get('sentences', []):
        if s.get('fakePercentage', 0) > 0:
            print(f"[{s.get('fakePercentage')}%] {s.get('sentence')}")
