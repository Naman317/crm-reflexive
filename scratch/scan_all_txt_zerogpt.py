import urllib.request
import json
import os
import glob
import time

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

def test_file(p):
    with open(p, 'r', encoding='utf-8') as f:
        txt = f.read().strip()
    if not txt:
        return
    payload = json.dumps({'input_text': txt}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            fake = data.get('fakePercentage', 0.0)
            words = data.get('textWords', 0)
            ai_w = data.get('aiWords', 0)
            print(f"{os.path.basename(p):<35} | Words: {words:>4} | AI: {fake:>5.1f}% | AI words: {ai_w}")
    except Exception as e:
        print(f"{os.path.basename(p):<35} | Error: {e}")

txt_files = glob.glob(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\*.txt')
for p in sorted(txt_files):
    test_file(p)
    time.sleep(1)
