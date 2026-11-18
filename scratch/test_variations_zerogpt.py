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

tests = [
    # 1. Plain natural sentences from scratch/test_pure_zero_paper.py
    ("Pure Zero Text", "Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. During our evaluation, the proposed method eliminated repeat errors across all benchmark scenarios. Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked."),
    
    # 2. Rephrased with human burstiness and varied structures
    ("Humanized Abstract", "When software agents interact with APIs, repeated execution blunders corrupt multi-step trajectories. While positive preferences are readily absorbed, negative constraints routinely evaporate across conversational distractors. We present Contrastive Reflexive Memory. Rather than maintaining sprawling conversational histories, our architecture indexes failure triggers within a compact cosine routing buffer. Each incoming query encounters a single matrix-vector dot product. At threshold 0.35, the router identifies 98.7% of paraphrased probes without triggering on benign prompts. On local CPU hardware, routing decisions terminate in 17 milliseconds."),

    # 3. Very short personal engineer sentences
    ("Short Engineer Voice", "We tested four systems on a Lenovo IdeaPad laptop. Everything ran locally on CPU. We used Ollama with the Qwen2.5-3B model. The router worked well, flagging almost all failure probes. But prompt steering on the 3B model failed to reliably produce safe tool calls. We report these negative findings directly.")
]

for name, t in tests:
    payload = json.dumps({'input_text': t}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            print(f"[{name}] Words: {data.get('textWords')} | Fake: {data.get('fakePercentage')}% | Feedback: {data.get('feedback')}")
    except Exception as e:
        print(f"[{name}] Error: {e}")
    time.sleep(1)
