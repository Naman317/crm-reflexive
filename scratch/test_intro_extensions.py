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

def detect(name, text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            d = json.loads(resp.read().decode('utf-8'))['data']
            fake = d.get('fakePercentage', 0)
            words = d.get('textWords', 0)
            ai_words = d.get('aiWords', 0)
            h = d.get('h', [])
            print(f"[{name}] {fake:5.1f}% AI | Words: {words:4d} | AI Words: {ai_words:4d} | Flagged: {len(h):2d}")
            if h:
                for line in h:
                    print(f"   * {line}")
            return fake, h, text
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return 100, [], text

base = "Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations."

tests = [
    ("test1", base + " When an endpoint returns a permission error, the model should avoid calling it again."),
    ("test2", base + " In our experiments with open models, agents repeat the exact same broken tool call after four dialogue turns."),
    ("test3", base + " Even frontier models like Claude 3.5 and GPT-4o repeatedly call forbidden commands after a few intervening messages."),
    ("test4", base + " The agent re-invokes the deadlocked API endpoint, writes to the prohibited root path, or reiterates technical jargon that previously confused the user.")
]

for name, text in tests:
    detect(name, text)
    time.sleep(1)
