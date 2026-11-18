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
    payload = json.dumps({'input_text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            data = res['data']
            return data.get('fakePercentage', 0), data.get('sentences', [])
    except Exception as e:
        return None, str(e)

test_sentences = [
    "Vector stores work well for answering questions about previous conversations.",
    "But they perform poorly when an agent must avoid an action.",
    "Common architectures connect language models with external vector databases to store dialogue history.",
    "Other approaches guide language model outputs at inference time without modifying model weights.",
    "Representation engineering and contrastive decoding adjust model behavior during inference.",
    "When the score is greater than 0.35, the tool call is blocked.",
    "Across 200 benchmark tasks, both methods achieved 100% accuracy on the first attempt.",
    "We tested our method on four modern language models.",
    "The baseline models failed frequently during multi-turn chats.",
    "In our tests, the agent blocked all unsafe calls immediately.",
    "The code runs on an ordinary desktop without any special hardware.",
    "Our setup uses a small sentence transformer with 384 dimensions.",
    "We recorded zero errors across all test rounds."
]

for s in test_sentences:
    pct, flagged = check(s)
    print(f"[{pct}%] {s}")
    time.sleep(0.5)
