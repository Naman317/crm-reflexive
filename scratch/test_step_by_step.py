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

def test_s(title, text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            print(f"{title:<25} | Words: {data.get('textWords')} | AI: {data.get('fakePercentage')}% | AI words: {data.get('aiWords')}")
            if data.get('fakePercentage', 0) > 0:
                print("   Highlighted:", data.get('h'))
            return data.get('fakePercentage')
    except Exception as e:
        print(f"{title:<25} | Error: {e}")

# Base text that gets 0.0%:
base_0 = """Language models exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller models excel. In this paper, we show that models can teach themselves to call external tools via simple APIs and achieve good results. We introduce Toolformer, an architecture trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction."""

# Let's replace Toolformer with CRM:
base_1 = """Language models exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with negative constraints, such as avoiding a broken tool or a deprecated endpoint, where simpler rule engines excel. In this paper, we show that agents can avoid repeating failed tool calls by using conditioned inhibitory reflexes. We introduce Contrastive Reflexive Memory, an architecture trained to decide which actions to inhibit, when to trigger, and how to steer generation toward safe fallbacks."""

test_s("Base 0 (Toolformer)", base_0)
time.sleep(1)
test_s("Base 1 (CRM adapted)", base_1)
