import urllib.request
import json
import time

def check_zerogpt(text, label=''):
    url = 'https://api.zerogpt.com/api/detect/detectText'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Origin': 'https://www.zerogpt.com',
        'Referer': 'https://www.zerogpt.com/'
    }
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            fake = data.get('fakePercentage', 0)
            words = data.get('textWords', 0)
            ai_words = data.get('aiWords', 0)
            h = data.get('h', [])
            print(f"[{label}] AI: {fake}% | Words: {words} | AI Words: {ai_words} | Flagged count: {len(h)}")
            if h:
                print("  Flagged:")
                for item in h:
                    print(f"    - {item}")
            return fake, h, words
    except Exception as e:
        print(f"[{label}] Error: {e}")
        return 100, [], 0

if __name__ == '__main__':
    print("ZeroGPT API verification harness ready.")
