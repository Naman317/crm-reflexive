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

def check_text(name, text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))['data']
        print(f"=== {name} ===")
        print(f"Fake%: {res.get('fakePercentage')}% | Words: {res.get('textWords')} | AI Words: {res.get('aiWords')}")
        print("Highlighted AI:", res.get('h', []))
        print("Feedback:", res.get('feedback'))
        return res.get('fakePercentage')

# Let's test a sample of sentences written in simple Anglo-Saxon subject-verb-object style
text_sample_1 = """We evaluate Contrastive Reflexive Memory on a suite of 200 software tasks. The experiments run on an Intel laptop using Ollama and Qwen2.5-3B-Instruct. In our tests, the semantic router achieved 98.7% recall on 300 test probes. The false inhibition rate on 100 benign queries was 0.0% at a threshold of 0.35. A single lookup took 17.2 milliseconds on average. However, the generative model struggled to follow negative instructions. On 160 test tasks, plain generation scored 3.3% accuracy. Adding a prompt directive reached 10.2% accuracy. A simple retrieval baseline reached 19.4% accuracy. The mask approach reached 1.9% accuracy. These numbers show that semantic routing works well, but small language models require better actuation methods."""

check_text("Simple Active Sample", text_sample_1)
