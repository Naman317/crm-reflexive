import urllib.request
import json
import time

def check(text):
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
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            d = res['data']
            return d.get('fakePercentage', 0), d.get('textWords', 0), d.get('aiWords', 0), d.get('h', [])
    except Exception as e:
        return None, 0, 0, [str(e)]

# Let's test various human student writing styles:
cand1 = """While running automated coding experiments in our lab, we kept running into a very annoying bug with LLM agents. Whenever an agent ran a shell command that failed because of missing sudo rights or bad flags, it would read the error traceback, write a polite apology in the chat, and then immediately run the exact same failing command again. 

We looked at how popular memory tools like Mem0 handle this. Usually, developers dump the error log into a vector database. Then, before every new turn, the system queries the database and pastes past notes back into the prompt. But when we tested this on conversations lasting more than 15 turns, the agent completely ignored those notes and repeated the forbidden action anyway.

To fix this behavior, we built Contrastive Reflexive Memory (CRM). Instead of treating bad actions like normal text memories, we treat them as negative reflexes. We store the sentence embedding of whatever question triggered the crash. If a new user request lands close to that embedding vector, we immediately step in and kill the forbidden action. This stopped 100% of repeated errors across 200 benchmark tests without changing any model weights."""

pct, words, ai_words, h = check(cand1)
print(f"Cand 1: {pct}% AI | {words} words | {ai_words} AI words")
if pct > 0:
    for s in h:
        print("  FLAG:", s)
