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

t1 = "Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. When an endpoint returns a permission error, the model should avoid calling it again."
t2 = t1 + " In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models."
t3 = t2 + " The embedding vector has 384 dimensions. When the score is greater than 0.35, the tool call is blocked."
t4 = t3 + " For open models, we set the logits of forbidden tool names to a large negative number before sampling."
t5 = t4 + " Across 200 benchmark tasks, both methods achieved 100% accuracy on the first attempt. In our experiments, the lookup runs in 0.15 milliseconds."

detect("t1", t1)
detect("t2", t2)
detect("t3", t3)
detect("t4", t4)
detect("t5", t5)
