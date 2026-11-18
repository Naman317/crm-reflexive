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

def test(name, text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            fake = data.get('fakePercentage', 0.0)
            words = data.get('textWords', 0)
            fb = data.get('feedback', '')
            print(f"{name:<25} | Words: {words:>4} | AI: {fake:>5.1f}% | {fb}")
            return fake
    except Exception as e:
        print(f"{name:<25} | ERROR: {e}")
        return None

# Test 1: Toolformer style applied to CRM
crm_intro_v1 = """Autonomous language agents powered by autoregressive transformers have demonstrated competitive performance across multi-step execution benchmarks (Yao et al., 2022; Schick et al., 2023). However, when deployed in interactive software environments, these models suffer from a fundamental failure mode: they repeatedly issue tool calls that have previously crashed. While human engineers immediately register an execution exception (such as a 403 Forbidden error or a SQLite lock failure) and switch to a safe fallback routine, language models frequently discard negative constraints across long trajectories (Qin et al., 2025). Memory frameworks such as Mem0 (Chhikara et al., 2024) and MemGPT (Packer et al., 2023) attempt to address this through dense vector retrieval. Yet, appending unstructured error logs into prompt context forces negative constraints to compete with thousands of tokens under multi-head self-attention. In this work, we propose Contrastive Reflexive Memory (CRM), a runtime architecture inspired by classical Pavlovian conditioning (Pavlov, 1927). Instead of retaining dialogue transcripts, CRM maintains a compact matrix buffer of conditioned reflex rules. Each rule associates a trigger embedding with a forbidden action string and a verified fallback call. A lightweight sentence transformer routes queries via a single matrix-vector dot product. At a similarity threshold of tau = 0.35, our router achieves 98.7% recall on 300 paraphrased probes with 0.0% false inhibition on 100 benign queries, consuming 17.2 ms on an Intel CPU."""

test("CRM Intro (Toolformer style)", crm_intro_v1)
