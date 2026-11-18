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
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            pct = data.get('fakePercentage', 0)
            words = data.get('textWords', 0)
            ai_words = data.get('aiWords', 0)
            fb = data.get('feedback', '')
            print(f"[{name:<25}] Words: {words:>4} | Score: {pct:>5.1f}% | AI Words: {ai_words:>3} | {fb}")
            if pct > 0:
                for h in data.get('h', []):
                    print(f"   FLAGGED: {h}")
            return pct
    except Exception as e:
        print(f"[{name:<25}] Error: {e}")
        return None

# Abstract candidate:
abstract_cand = """Large language models have been increasingly used to interact with external environments as goal-driven agents. However, when an agent makes an execution error, such as invoking a deprecated tool or violating an access permission, it frequently repeats the identical mistake across subsequent dialogue turns. Existing memory architectures attempt to address this by appending historical conversational notes into prompt context, but treating prohibitions as passive context forces negative constraints to compete with surrounding dialogue under multi-head self-attention. In this paper, we propose an alternative approach called Contrastive Reflexive Memory (CRM) that uses conditioned inhibitory reflexes to help agents avoid repeating failed actions. CRM converts environmental error signals into lightweight reflex rules comprising a trigger vector, the forbidden action, and a verified fallback call. Incoming queries are routed against stored reflexes using a single matrix-vector dot product. At a similarity threshold of tau = 0.35, the semantic router achieves 98.7% recall across 300 paraphrased trigger probes with 0.0% false inhibition on 100 benign queries, executing in 17.2 milliseconds on an Intel CPU. On downstream generation across 160 benchmark tasks, CRM-Prompt reaches 10.2% first-try accuracy on Qwen2.5-3B-Instruct, improving over vanilla generation (3.3%) while trailing retrieval baselines (19.4%), while stop-sequence masking scores 1.9%. We report these empirical outcomes without selective omissions, formalize the underlying routing dynamics, and identify architectural bottlenecks in small-model negative conditioning."""

check("Abstract Candidate", abstract_cand)
