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
                for line in d.get('h', []):
                    print(f"   FLAGGED: {line}")
            return pct
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

b3 = """Standard language models learn positive preferences easily, but fail to remember negative constraints. To quantify this failure, we compute the First-Try Accuracy on the test prompt using the indicator metric defined in (4). We define Delta as the accuracy difference between positive and negative tasks. Over longer dialogues, models repeatedly invoke deprecated actions. In Figure 2, we show how paraphrase variations cause retrieval baselines to degrade across syntactic tiers."""

check("B3", b3)
