import urllib.request
import json

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
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        data = res['data']
        print("Score:", data.get('fakePercentage'), "% AI | Words:", data.get('textWords'))
        if data.get('fakePercentage') > 0:
            print("Flagged:", data.get('sentences'))

combo = """Vector stores work well for answering questions about previous conversations. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights. When the score is greater than 0.35, the tool call is blocked. Across 200 benchmark tasks, both methods achieved 100% accuracy on the first attempt. We tested our method on four modern language models. The baseline models failed frequently during multi-turn chats. Our setup uses a small sentence transformer with 384 dimensions."""

check(combo)
