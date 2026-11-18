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

def check(text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode('utf-8')).get('data', {})
                return {
                    'score': data.get('fakePercentage', 0),
                    'words': data.get('textWords', 0),
                    'ai_words': data.get('aiWords', 0),
                    'feedback': data.get('feedback', ''),
                    'sentences': [s for s in data.get('sentences', []) if s.get('fakePercentage', 0) > 0]
                }
        except Exception:
            time.sleep(1)
    return None

# Let's inspect the sections we have
print("Test script ready.")
