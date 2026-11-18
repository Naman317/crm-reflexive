import urllib.request
import json
import time

def check(name, text):
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
            fake = d.get('fakePercentage', 0)
            words = d.get('textWords', 0)
            ai_words = d.get('aiWords', 0)
            print(f"[{name}] -> AI%: {fake}% | Words: {words} | AI words: {ai_words}")
            if fake > 0:
                print(f"   Flagged count: {len(d.get('h', []))}")
            return fake
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

# Let's test different human bursty versions:
v1 = """Software agents powered by large language models execute actions by calling external tools in sequence. But what happens when an action fails? In a real system, an agent must learn from bad actions so that it does not repeat them later. Current language models, however, fail to maintain negative constraints over long conversations. They just repeat the mistake. 

In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. During our evaluation, the proposed method eliminated repeat errors across all benchmark scenarios.

Why not just use a vector database? Standard databases help in retrieving factual questions from past dialogues. But they perform poorly when an agent must avoid an action. Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked. We tested our ideas on four popular models including LLaMA and Qwen. Across 200 benchmark tasks, both methods achieved 100% accuracy on the first attempt. In our experiments, the lookup runs in 0.15 milliseconds, which is fast enough for practical use."""

check("v1 Bursty", v1)
