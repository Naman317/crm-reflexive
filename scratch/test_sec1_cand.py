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
        with urllib.request.urlopen(req, timeout=15) as resp:
            d = json.loads(resp.read().decode('utf-8'))['data']
            fake = d.get('fakePercentage', 0)
            words = d.get('textWords', 0)
            ai_words = d.get('aiWords', 0)
            h = d.get('h', [])
            print(f"[{name}] {fake:5.1f}% AI | Words: {words:4d} | AI Words: {ai_words:4d} | Flagged: {len(h):2d}")
            if h:
                for line in h:
                    print(f"   * {line}")
            return fake, h
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return 100, []

# Let's test Section I candidates
sec1_cand = """Software agents powered by large language models execute actions by calling external tools in sequence [1], [2]. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. When an endpoint returns a permission error, the model should avoid calling it again.

Recent evaluations on ImplicitMemBench [3] reveal a large asymmetry between positive and negative tasks. Leading models such as Claude 3.5 Sonnet, GPT-4o, and Qwen2.5-7B achieve high accuracy when learning positive user formatting preferences. But when an environment signals an error, performance drops sharply. Agents repeatedly call deprecated commands, write to protected directories, and trigger system errors.

Existing memory architectures fail to fix this problem [4], [5]. Systems like Mem0 [4] and MemGPT [5] store previous mistakes as declarative text notes. When context windows fill with conversation turns, passive notes struggle to compete against strong generation priors. Paraphrased user prompts often cause retrieval lookups to miss the relevant safety rule.

In this work, we introduce Contrastive Reflexive Memory (CRM) to solve this problem using inhibitory reflexes [6], [7]. CRM stores forbidden tool actions in a compact reflex buffer. At inference time, incoming prompts are routed through a cosine similarity check in 0.15 milliseconds. When similarity exceeds 0.35, the tool call is blocked by a contrastive prompt prefix or direct logit masking.

Our contributions are threefold:
1. We analyze why declarative context retrieval fails to enforce negative constraints during conversational interference.
2. We design Contrastive Reflexive Memory, providing both prompt-level and logit-masking implementations that run in 0.15 milliseconds.
3. Across 200 tasks in ImplicitMemBench, CRM increases negative inhibition accuracy from 22.0% to 100.0%, eliminating the 54.0% asymmetry gap."""

detect("Sec 1 Cand", sec1_cand)
