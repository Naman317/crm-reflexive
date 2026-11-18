from pypdf import PdfReader
import urllib.request
import json
import time

pdfs = [
    ("ieee_double_column_conference.pdf", r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_conference.pdf'),
    ("ieee_double_column_blind.pdf", r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_blind.pdf'),
    ("ieee_single_column_journal.pdf", r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_single_column_journal.pdf')
]

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

for name, path in pdfs:
    reader = PdfReader(path)
    extracted = "\n\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    payload = json.dumps({'input_text': extracted}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            print(f"[{name}]")
            print(f"  Pages: {len(reader.pages)} | Characters: {len(extracted)} | Words: {data.get('textWords')}")
            print(f"  AI %: {data.get('fakePercentage')}% | AI Words: {data.get('aiWords')}")
            print(f"  Feedback: {data.get('feedback')}")
    except Exception as e:
        print(f"[{name}] Error: {e}")
    time.sleep(1)
