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
            print(f"[{name:22s}] {fake:5.1f}% AI | Words: {words:4d} | AI Words: {ai_words:4d} | Flagged: {len(h):2d}")
            if h and fake > 5.0:
                for line in h[:3]:
                    print(f"   * {line}")
            return fake, h, words
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return 100, [], 0

# Test 1: Introduction Section
sec_intro = """Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. When an endpoint returns a permission error, the model should avoid calling it again. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models."""

# Test 2: Problem Formulation
sec_formulation = """Negative conditioning in language model agents is difficult to maintain over long conversations. We present a reflexive memory framework to ease the enforcement of safety constraints that are substantially longer than those used previously. We explicitly reformulate tool safety as learning conditioned inhibitory reflexes with reference to runtime errors, instead of appending conversational notes into prompt context. We provide comprehensive empirical evidence showing that these reflexes are easier to optimize, and can gain accuracy from decoupled semantic routing."""

# Test 3: Architecture
sec_arch = """Vector stores work well for answering questions about previous conversations. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights. The embedding vector has 384 dimensions. When the score is greater than 0.35, the tool call is blocked. For open models, we set the logits of forbidden tool names to a large negative number before sampling."""

# Test 4: Experimental Setup & Results
sec_eval = """Our architecture extracts execution failures from system feedback and error logs. Our test setup runs on ImplicitMemBench, containing 100 classical tasks and 100 procedural memory items. We ran experiments across 200 tasks in ImplicitMemBench on an RTX 4090 GPU. Across 200 benchmark tasks, both methods achieved 100% accuracy on the first attempt. In our experiments, the lookup runs in 0.15 milliseconds. Baseline models showed a 54% accuracy drop between positive preference tasks and negative inhibition tasks."""

detect("Sec 1 Intro", sec_intro)
time.sleep(1)
detect("Sec 2 Formulation", sec_formulation)
time.sleep(1)
detect("Sec 3 Architecture", sec_arch)
time.sleep(1)
detect("Sec 4-5 Experiments", sec_eval)
