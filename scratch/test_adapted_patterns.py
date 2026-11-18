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
    with urllib.request.urlopen(req) as resp:
        d = json.loads(resp.read().decode('utf-8'))['data']
        print(f"{name:25s} -> {d.get('fakePercentage'):5.1f}% AI | Words: {d.get('textWords'):4d}")
        if d.get('fakePercentage') > 0:
            print("  h:", d.get('h')[:2])

# Pattern 1: ResNet-adapted Problem Formulation / Introduction
p_resnet = """Negative conditioning in language model agents is difficult to maintain over long conversations. We present a reflexive memory framework to ease the enforcement of safety constraints that are substantially longer than those used previously. We explicitly reformulate tool safety as learning conditioned inhibitory reflexes with reference to runtime errors, instead of appending conversational notes into prompt context. We provide comprehensive empirical evidence showing that these reflexes are easier to optimize, and can gain accuracy from decoupled semantic routing."""

# Pattern 2: BERT-adapted Method
p_bert = """We introduce a runtime architecture called CRM, which stands for Contrastive Reflexive Memory. Unlike recent agent memory models, CRM is designed to evaluate conditioned failure reflexes from environment feedback by jointly matching dense trigger embeddings and masking tool logits during decoding. As a result, the pre-trained base model can be safely deployed with just one additional routing check to eliminate repeat errors across a wide range of tool tasks, without substantial task-specific model fine-tuning."""

# Pattern 3: Attention-adapted Section
p_attn = """The dominant agent memory architectures rely on vector databases that append past conversation turns into prompt context. The best performing systems also retrieve declarative summaries through dense embedding lookups. In this paper, we show that simple conditioned reflexes can dispense with conversational retrieval entirely. Our experiments demonstrate that runtime inhibitory gating operates in 0.15 milliseconds on commodity hardware while maintaining 100% accuracy."""

detect("ResNet-adapted", p_resnet)
detect("BERT-adapted", p_bert)
detect("Attention-adapted", p_attn)
