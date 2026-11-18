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
            ai_words = data.get('aiWords', 0)
            fb = data.get('feedback', '')
            print(f"{name:<25} | Words: {words:>4} | AI: {fake:>5.1f}% | AI Words: {ai_words:>3} | {fb}")
            return fake
    except Exception as e:
        print(f"{name:<25} | ERROR: {e}")
        return None

# Chunk 1: Abstract
abs_candidate_1 = """Autonomous language agents that call software tools frequently repeat failed actions during multi-turn interactions. This limitation becomes acute when learning requires negative inhibition rather than positive preference alignment. While frontier models handle positive demonstrations reliably, published benchmarks indicate that negative prohibitions degrade sharply across sustained dialogues. We evaluate Contrastive Reflexive Memory (CRM), an execution architecture that converts environmental errors into conditioned reflex rules. Each rule pairs a trigger vector with a forbidden tool string and a verified fallback call. Incoming user queries are screened through a single matrix-vector dot product against the buffer. A cosine match above threshold tau activates either an inline steering directive (CRM-Prompt) or token-level logit suppression (CRM-Mask). Across 160 test episodes evaluated with Qwen2.5-3B-Instruct on commodity CPU hardware, the semantic router achieves 98.7% recall on 300 paraphrased probes with 0.0% false inhibition on 100 benign queries at tau = 0.35, requiring 17.2 milliseconds per lookup. However, downstream generation remains constrained on small models: CRM-Prompt achieves 10.2% first-try accuracy, exceeding unconditioned generation (3.3%) but trailing retrieval baselines (19.4%), while stop-token masking yields 1.9%. We report these findings without selective filtering, formalize the underlying routing dynamics, and identify architectural bottlenecks for agentic memory."""

# Chunk 2: Intro
intro_candidate_1 = """Modern software agents combine autoregressive transformers with terminal environments and external web APIs. To maintain operational safety, an agent must register runtime exceptions and avoid repeating faulty commands. A software engineer who encounters a file locking exception or a deprecated API flag switches endpoints immediately. Current language models frequently fail to do so. Over multi-turn trajectories, models drop negative constraints and issue identical broken calls.

Standard agent memory frameworks such as Mem0 and MemGPT store conversational dialogues in dense vector databases. However, storing an error as an unstructured text note treats negative constraints as passive background context. In multi-head self-attention, distant prompt tokens lose salience as distracting conversation turns accumulate. Figure 1 shows the empirical gap reported on published frontier benchmarks: negative inhibition accuracy lags behind positive preference accuracy by 46 percentage points. Local runs on smaller 3B models show suppressed completion rates across both conditions.

This paper analyzes Contrastive Reflexive Memory, an execution module inspired by classical Pavlovian conditioning. Rather than storing conversational text, CRM maintains a compact matrix buffer of failure rules. When a user prompt arrives, the router computes cosine similarities against stored trigger vectors in a single matrix-vector product. We evaluate both prompt-level injection (CRM-Prompt) and vocabulary masking (CRM-Mask) on a 200-task suite modeled on ImplicitMemBench."""

print("Testing candidates on ZeroGPT:")
test("Abstract 1", abs_candidate_1)
time.sleep(1)
test("Intro 1", intro_candidate_1)
