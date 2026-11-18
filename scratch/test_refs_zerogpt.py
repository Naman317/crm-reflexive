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

papers = {
    'MemGPT': r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\reference_papers\MemGPT_Packer_2023.pdf',
    'Toolformer': r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\reference_papers\Toolformer_Schick_2023.pdf',
    'ReAct': r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\reference_papers\ReAct_Yao_2022.pdf',
    'Reflexion': r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\reference_papers\Reflexion_Shinn_2023.pdf',
    'Mem0': r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\reference_papers\Mem0_Chhikara_2024.pdf',
}

for name, path in papers.items():
    reader = PdfReader(path)
    text = reader.pages[0].extract_text()
    # take intro or abstract
    sample = text[:1500]
    payload = json.dumps({'input_text': sample}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))['data']
        print(f"{name:<12} | Words: {res.get('textWords')} | AI: {res.get('fakePercentage')}% | {res.get('feedback')}")
    time.sleep(1)
