import urllib.request
import json
from pypdf import PdfReader

pdf_path = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_layout_balanced.pdf'
reader = PdfReader(pdf_path)

extracted_pages = []
for i, page in enumerate(reader.pages):
    txt = page.extract_text()
    if txt:
        extracted_pages.append(txt)

full_extracted = "\n\n".join(extracted_pages)
print(f"Total pages: {len(reader.pages)}")
print(f"Total extracted characters: {len(full_extracted)}")
print(f"Total extracted words: {len(full_extracted.split())}")
print("-" * 50)
print(full_extracted[:300])
print("-" * 50)

# Query ZeroGPT
url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

payload = json.dumps({'input_text': full_extracted}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print("\n================== ZEROGPT PDF AUDIT ==================")
    print(f"AI Percentage: {data.get('fakePercentage')}%")
    print(f"Total Words: {data.get('textWords')}")
    print(f"AI Words: {data.get('aiWords')}")
    print(f"Feedback: {data.get('feedback')}")
    print("=======================================================")
