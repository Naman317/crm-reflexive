import urllib.request
import json
import time
from pypdf import PdfReader

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

reader = PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\reference_papers\Toolformer_Schick_2023.pdf')
text = reader.pages[0].extract_text()
paras = [p.strip() for p in text.split('\n\n') if len(p.strip().split()) > 30]

for i, p in enumerate(paras[:4]):
    payload = json.dumps({'input_text': p}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))['data']
        print(f"Para {i+1} | Words: {res.get('textWords')} | AI: {res.get('fakePercentage')}% | {res.get('feedback')}")
        print("Text:", p[:150].replace('\n', ' '))
        print('-'*40)
    time.sleep(1)
