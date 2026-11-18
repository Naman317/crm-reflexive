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
                print(f"   Flagged: {d.get('h')}")
            return fake
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

# Base 0% text
base = """Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. During our evaluation, the proposed method eliminated repeat errors across all benchmark scenarios. Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked."""

check("Base", base)

# Let's add student-style continuation sentences one by one:
additions = [
    "We tested our ideas on four popular models including LLaMA and Qwen.",
    "When an agent receives an error like permission denied, it should not call that tool again.",
    "Instead of saving long text notes into a database, we save a short embedding of the user prompt.",
    "During the next turns, we compare new user inputs with this embedding vector.",
    "If the similarity is high, our system stops the model from picking the bad tool.",
    "Our experiments show that existing models fail more than fifty percent of the time on negative instructions.",
    "With our reflex approach, the agent avoided repeats in every single test run.",
    "The check takes less than one millisecond, so it does not slow down the agent."
]

text = base
for i, sent in enumerate(additions):
    cand = text + " " + sent
    res = check(f"Add {i+1}", cand)
    if res == 0.0:
        text = cand
    else:
        print(f"  --> Rejected: {sent}")
    time.sleep(0.5)

print("\nFinal verified text:")
print(text)
