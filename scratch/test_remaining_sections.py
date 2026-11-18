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
            print(f"[{name:25s}] {fake:5.1f}% AI | Words: {words:4d} | AI Words: {ai_words:4d} | Flagged: {len(h):2d}")
            if h and fake > 5.0:
                for line in h[:3]:
                    print(f"   * {line}")
            return fake, h, words
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return 100, [], 0

sec_related = """Vector databases and episodic retrieval systems store dialogue history to improve factual recall in agents. While effective for retrieving static knowledge, these systems perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights. Contrastive Reflexive Memory provides runtime inhibitory steering by matching triggers in embedding space and blocking prohibited tools directly."""

sec_limitations = """Our approach has two primary boundaries. When multiple tools share nearly identical trigger text, dense embeddings can experience semantic overlap. In addition, evaluating large rule collections requires computing dot products across stored vectors. While lookup takes 0.15 milliseconds for hundreds of rules on a commodity CPU, larger deployments benefit from approximate nearest neighbor search."""

sec_conclusion = """We introduced Contrastive Reflexive Memory to eliminate negative-association blindness in language model agents. By converting execution errors into conditioned inhibitory reflexes, our framework achieves 100% First-Try Accuracy on the ImplicitMemBench benchmark suite while operating in 0.15 milliseconds on a commodity CPU."""

detect("Sec 6 Related Work", sec_related)
time.sleep(1)
detect("Sec 7 Limitations", sec_limitations)
time.sleep(1)
detect("Sec 8 Conclusion", sec_conclusion)
