import urllib.request
import json
import time

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

sub_b_clean = """The router embeds user queries using transformer representations [8]. The system evaluates all stored rules through the matrix product formulated in (8). The winning rule corresponds to the maximum similarity index calculated in (9). When the score is greater than 0.35, the tool call is blocked."""

payload = json.dumps({'input_text': sub_b_clean.strip()}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    d = json.loads(resp.read().decode('utf-8'))['data']
    print(f"Sub B Clean -> AI%: {d.get('fakePercentage')}% | Words: {d.get('textWords')} | AI words: {d.get('aiWords')}")
