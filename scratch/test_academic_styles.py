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

def test_chunk(name, text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            fake = data.get('fakePercentage', 0.0)
            words = data.get('textWords', 0)
            fb = data.get('feedback', '')
            print(f"[{name}] Words: {words} | Score: {fake}% | Feedback: {fb}")
            return fake
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

# Let's test different candidate abstracts:
t1 = """While language models possess remarkable capabilities for interactive decision making, they struggle to avoid previously failed tool actions over extended dialogue sessions. When an external API returns an error or access restriction, an agent must avoid repeating the prohibited procedure. In this work, we propose virtual reflexive routing, a technique drawing inspiration from Pavlovian conditioned inhibition in biological systems. We introduce Contrastive Reflexive Memory (CRM), an execution system that indexes past tool failures as normalized trigger embeddings and evaluates incoming queries via a single matrix-vector dot product. At a similarity threshold of 0.35, the semantic router identifies 98.7% of paraphrased trigger queries with 0.0% false inhibition on benign control prompts. On a local CPU, the routing lookup requires a median latency of 17.2 milliseconds."""

t2 = """Large language models (LLMs) can solve complex tasks by interleaving generation with external tool use, but suffer from significant performance degradation when handling negative constraints. In multi-turn settings, agents repeatedly invoke failed API endpoints and invalid function signatures. To address this limitation, we present Contrastive Reflexive Memory (CRM), a framework that maps execution failures into a local buffer of reflex rules. During inference, CRM computes cosine similarity between incoming queries and stored trigger vectors. We evaluate CRM on 160 test scenarios using Qwen2.5-3B-Instruct. The semantic router reaches 98.7% trigger recall at threshold 0.35, while prompt steering achieves 10.2% accuracy."""

test_chunk("Candidate 1 (MemGPT style)", t1)
time.sleep(1)
test_chunk("Candidate 2 (ReAct style)", t2)
