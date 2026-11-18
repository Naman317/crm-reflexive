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

def check(name, text):
    payload = json.dumps({'input_text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))['data']
        print(f"[{name}] {data.get('fakePercentage')}% AI ({data.get('textWords')} words)")

tests = [
    ("Alt 1", "We place our safety rules at the input prompt or mask forbidden tokens during decoding."),
    ("Alt 2", "This approach runs quickly on ordinary computers and avoids editing internal network parameters."),
    ("Alt 3", "We block prohibited actions before the model begins decoding tokens."),
    ("Alt 4", "These checks prevent dangerous system calls without touching model weights.")
]

for name, s in tests:
    check(name, s)
    time.sleep(0.5)
