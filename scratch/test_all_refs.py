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

def check(name, text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            print(f"{name:<20} | Words: {data.get('textWords')} | Score: {data.get('fakePercentage')}% | {data.get('feedback')}")
            return data.get('fakePercentage')
    except Exception as e:
        print(f"{name:<20} | Error: {e}")

papers = {
    'Toolformer P2': (r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\reference_papers\Toolformer_Schick_2023.pdf', 1),
    'ReAct P2': (r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\reference_papers\ReAct_Yao_2022.pdf', 1),
    'Reflexion P2': (r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\reference_papers\Reflexion_Shinn_2023.pdf', 1),
    'MemGPT P2': (r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\reference_papers\MemGPT_Packer_2023.pdf', 1),
}

for name, (path, p_idx) in papers.items():
    reader = PdfReader(path)
    text = reader.pages[p_idx].extract_text()
    check(name, text[:1500])
    time.sleep(1)
