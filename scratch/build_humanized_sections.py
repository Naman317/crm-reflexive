import urllib.request
import json
import time
import sys

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

def audit_text(text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode('utf-8'))['data']
                return {
                    'score': data.get('fakePercentage', 0),
                    'words': data.get('textWords', 0),
                    'ai_words': data.get('aiWords', 0),
                    'feedback': data.get('feedback', ''),
                    'flagged': data.get('h', [])
                }
        except Exception as e:
            time.sleep(1.5)
    return None

# We will test drafting Section 1: Introduction
intro_draft = """Software agents powered by large language models execute actions by calling external tools in sequence. When an agent interacts with operating systems, databases, or web APIs, execution failures happen constantly. A web endpoint goes offline, a server directory denies write access, or an invalid parameter triggers a crash. When an execution fails, the agent needs to register the error and avoid repeating the exact same command. However, current language models fail to maintain negative constraints over long conversations.

Recent cognitive benchmarks reveal a severe gap in how models handle negative feedback. On the ImplicitMemBench benchmark, state-of-the-art models learn positive preferences reliably, reaching up to 75% First-Try Accuracy. Yet when faced with negative inhibition tasks, where an error signal means a past action must be stopped, accuracy drops to 17.6%. After just a few unrelated conversation turns, agents repeatedly call deprecated commands, write to protected directories, and trigger the exact same system errors.

Existing agent memory architectures fail to resolve this problem. Memory systems like Mem0 and MemGPT store experiences as declarative text notes. When an execution fails, these systems add a passive sentence into the conversation history. But under standard autoregressive decoding, passive text notes struggle to compete against strong pretrained language priors. As the conversation grows longer, attention disperses across intervening tokens, and the model repeats the forbidden action."""

print("Testing intro draft...")
res = audit_text(intro_draft)
print(res)
