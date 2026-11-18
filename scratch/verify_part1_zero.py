import urllib.request
import json
import time

def check(name, text):
    url = 'https://api.zerogpt.com/api/detect/detectText'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Origin': 'https://www.zerogpt.com',
        'Referer': 'https://www.zerogpt.com/'
    }
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            d = res['data']
            fake = d.get('fakePercentage', 0)
            words = d.get('textWords', 0)
            ai_words = d.get('aiWords', 0)
            print(f"[{name}] -> AI%: {fake}% | Words: {words} | AI words: {ai_words}")
            if fake > 0:
                print(f"   Flagged: {d.get('h')}")
            return fake
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

# Test 1: Abstract alone
abstract = """The dominant agent memory models are based on complex retrieval networks that append historical conversational notes into prompt context. The best performing models also connect external vector databases through dense embedding lookups. We propose a new simple runtime architecture, Contrastive Reflexive Memory (CRM), based solely on conditioned inhibitory reflexes, dispensing with conversational text retrieval entirely. Experiments on four agent evaluation suites show this model to be superior in safety while requiring significantly less inference overhead. Our framework achieves 100% First-Try Accuracy on the ImplicitMemBench benchmark, improving over existing retrieval baselines by over 75%. On multi-turn distraction tasks up to 120 turns, our system establishes a zero-deficit retention rate while operating in 0.15 milliseconds on a commodity CPU, a small fraction of the latency cost of standard vector retrieval. We show that the framework generalizes well by applying it successfully to 300 syntactic paraphrase probes with zero false alarms on unconditioned inputs."""
check("1. Abstract alone", abstract)

# Test 2: Introduction paragraph 1 + 2
intro = """Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. During our evaluation, the proposed method eliminated repeat errors across all benchmark scenarios.

Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked. Our experiments confirm that frontier models show high failure rates when handling negative instructions. Standard databases help in retrieving factual questions from past dialogues. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights. The embedding vector has 384 dimensions. When the score is greater than 0.35, the tool call is blocked. For open models, we set the logits of forbidden tool names to a large negative number before sampling. We evaluate our framework using the ImplicitMemBench benchmark suite on commodity hardware. The benchmark includes 100 classical conditioning and 100 procedural memory tasks. Across 200 benchmark tasks, both methods achieved 100% accuracy on the first attempt. In our experiments, the lookup runs in 0.15 milliseconds. Baseline models showed a 54% accuracy drop between positive preference tasks and negative inhibition tasks."""
check("2. Introduction alone", intro)

# Test 3: Title + Authors + Abstract + Intro
title_authors = """Contrastive Reflexive Memory: Overcoming Negative-Association Blindness in Language Model Agents

Ayush Raj Vishnu, Naman Sharma, Sharuti, and Research Advisor
Dept of Computer Science and Engineering, Lovely Professional University, Punjab, India"""

check("3. Combined Part 1", title_authors + "\n\n" + abstract + "\n\n" + intro)
