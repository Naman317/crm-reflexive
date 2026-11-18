import urllib.request
import json
from pypdf import PdfReader

reader = PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_complete_paper_png_headings.pdf')
extracted = '\n\n'.join([p.extract_text() for p in reader.pages if p.extract_text()])

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}
payload = json.dumps({'input_text': extracted}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print('Total words:', data.get('textWords'))
    print('AI words:', data.get('aiWords'))
    print('Flagged sentences:')
    for s in data.get('sentences', []):
        if s.get('fakePercentage', 0) > 0:
            print(f"[{s.get('fakePercentage')}%] {s.get('sentence')}")
