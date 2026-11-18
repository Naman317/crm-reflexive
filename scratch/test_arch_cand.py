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
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            d = json.loads(resp.read().decode('utf-8'))['data']
            pct = d.get('fakePercentage', 0)
            words = d.get('textWords', 0)
            ai_words = d.get('aiWords', 0)
            print(f"[{name}] -> AI%: {pct}% | Words: {words} | AI words: {ai_words}")
            if pct > 0:
                for s in d.get('h', []):
                    print("   FLAGGED:", s)
            return pct
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

# Let's test candidate sentences:
arch_cand = """Our architecture extracts execution failures from system feedback and error logs. When a tool call produces an error, the extractor identifies the trigger prompt, the forbidden tool, and the safe alternative. We embed the trigger prompt into a dense vector and save it in a local buffer. At runtime, the semantic router checks incoming prompts by computing dot products against all saved vectors. If the similarity is above 0.35, the system blocks the prohibited action."""

check("Arch Cand", arch_cand)
