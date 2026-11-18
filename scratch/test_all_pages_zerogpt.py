import re
import urllib.request
import json
import time
from pypdf import PdfReader

reader = PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\reflexive_inhibition_paper.pdf')

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

def clean_txt(t):
    t = re.sub(r'(?<!\n)\n(?!\n)', ' ', t)
    t = t.replace('ﬀ', 'ff').replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('ﬃ', 'ffi').replace('–', '-').replace('—', '-')
    return re.sub(r'\s+', ' ', t).strip()

page_texts = [clean_txt(p.extract_text()) for p in reader.pages if p.extract_text()]

print("Testing each page on ZeroGPT:")
total_words = 0
total_ai_words = 0

for i, pt in enumerate(page_texts):
    payload = json.dumps({'input_text': pt}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            pct = data.get('fakePercentage', 0)
            words = data.get('textWords', 0)
            ai = data.get('aiWords', 0)
            total_words += words
            total_ai_words += ai
            print(f"Page {i+1} | Words: {words:>4} | AI: {pct:>5.1f}% | AI words: {ai:>3} | {data.get('feedback')}")
    except Exception as e:
        print(f"Page {i+1} | Error: {e}")
    time.sleep(1)

print("-" * 50)
if total_words > 0:
    overall_pct = (total_ai_words / total_words) * 100
    print(f"OVERALL: Total Words = {total_words}, Total AI Words = {total_ai_words}, Overall AI% = {overall_pct:.2f}%")
