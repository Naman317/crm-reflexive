from pypdf import PdfReader
import urllib.request
import json

reader = PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_sections_compile.pdf')
txt = "\n\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
print(f"Extracted length: {len(txt)} chars | {len(txt.split())} words")
print("-" * 50)
print(txt[:300])
print("-" * 50)

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

payload = json.dumps({'input_text': txt}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode('utf-8'))
    print("Full API response keys:", res.keys())
    if 'data' in res and res['data']:
        d = res['data']
        print(f"Score: {d.get('fakePercentage')}% AI | Words: {d.get('textWords')} | AI words: {d.get('aiWords')}")
        print(f"Feedback: {d.get('feedback')}")
        print(f"Flagged: {d.get('sentences')}")
    else:
        print("API message:", res.get('message'))
