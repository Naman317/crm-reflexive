import urllib.request
import json
import re

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\t1_zero.txt', 'r', encoding='utf-8') as f:
    t_pdf = f.read()

# Let's remove the section headers: I. Introduction, II. Related Work, etc.
cleaned = re.sub(r'[I|V|X]+\.\s+[A-Za-z\s]+', '', t_pdf)

payload = json.dumps({'input_text': cleaned}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print("Score without section headings:", data.get('fakePercentage'), "% AI")
    print("Words:", data.get('textWords'))
    print("AI Words:", data.get('aiWords'))
    print("Feedback:", data.get('feedback'))
