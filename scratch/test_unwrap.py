import urllib.request
import json
import re
from pypdf import PdfReader

reader = PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_clean_nohyphen.pdf')
raw_extracted = '\n\n'.join([p.extract_text() for p in reader.pages if p.extract_text()])

# Un-wrap lines within paragraphs (join lines that don't end with double newline)
unwrapped = re.sub(r'(?<!\n)\n(?!\n)', ' ', raw_extracted)

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

payload = json.dumps({'input_text': unwrapped}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print("Unwrapped score:", data.get('fakePercentage'), "% AI")
    print("Words:", data.get('textWords'))
    print("AI Words:", data.get('aiWords'))
    print("Feedback:", data.get('feedback'))
