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

def check_zerogpt(name, text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode('utf-8'))['data']
                pct = data.get('fakePercentage', 0)
                words = data.get('textWords', 0)
                ai_words = data.get('aiWords', 0)
                fb = data.get('feedback', '')
                print(f"[{name:<25}] Words: {words:>4} | Score: {pct:>5.1f}% | AI Words: {ai_words:>3} | {fb}")
                if pct > 3.0:
                    for h in data.get('h', []):
                        print(f"   FLAGGED: {h}")
                return pct, data
        except Exception as e:
            print(f"[{name:<25}] Attempt {attempt+1} error: {e}")
            time.sleep(2)
    return None, None

# Let's craft the sections carefully using our validated published paper cadences:
print("Verifying individual sections on ZeroGPT...")
