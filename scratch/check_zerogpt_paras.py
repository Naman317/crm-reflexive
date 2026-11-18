import urllib.request
import json

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\extracted_paper_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip().split()) > 25]

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/',
}

for i, p in enumerate(paragraphs[:8]):
    payload = json.dumps({'input_text': p}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            res = json.loads(resp.read().decode('utf-8'))['data']
            fake = res.get('fakePercentage', 0)
            ai_w = res.get('aiWords', 0)
            print(f"Para {i+1} ({len(p.split())} words): AI = {fake}% | flagged = {ai_w}")
            if fake > 0:
                print("   Preview:", p[:100], "...")
    except Exception as e:
        print('Error:', e)
