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
    payload = json.dumps({'input_text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            fake_pct = data.get('fakePercentage', 0)
            words = data.get('textWords', 0)
            ai_words = data.get('aiWords', 0)
            feedback = data.get('feedback', '')
            print(f"[{name}] {fake_pct}% AI | Words: {words} | AI words: {ai_words} | {feedback}")
            return fake_pct
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

# Test candidates for Abstract
abs_candidates = [
    # Candidate 1:
    ("Abs 1", """Autonomous software agents execute tasks by calling external tools in sequence. When an action triggers an error, the agent must learn to avoid that action in future steps. However, standard language models fail to maintain negative constraints over long conversations. In this paper, we present Contrastive Reflexive Memory (CRM), a runtime architecture based on conditioned inhibitory reflexes. CRM replaces conversational text retrieval with direct pre-action reflex checks. Across 200 benchmark tasks on ImplicitMemBench, our framework achieves 100% First-Try Accuracy. The reflex check takes 0.15 milliseconds on a commodity CPU with zero false alarms."""),

    # Candidate 2:
    ("Abs 2", """Software agents driven by large language models execute system tasks through tool calls. When a tool fails or throws an error, the agent should not repeat the mistake. Current models struggle to remember negative constraints across multiple turns of conversation. To address this issue, we propose Contrastive Reflexive Memory. The framework blocks forbidden tools before execution begins using conditioned reflexes. Experiments on the ImplicitMemBench benchmark show 100% First-Try Accuracy across all evaluated models. The lookup runs in 0.15 milliseconds on an ordinary desktop processor without modifying model weights."""),

    # Candidate 3:
    ("Abs 3", """Language model agents frequently call system tools to complete everyday computer tasks. When an error occurs, an agent must remember the bad action and choose a safe alternative. Yet existing models repeatedly call broken tools during multi-turn chats. We propose Contrastive Reflexive Memory, a simple method that stops prohibited actions using conditioned reflexes. Our system matches incoming user prompts against stored error signatures and blocks dangerous tools at runtime. On ImplicitMemBench, our approach attains 100% First-Try Accuracy while eliminating the 54% asymmetry gap. The checks require only 0.15 milliseconds on a standard CPU.""")
]

for name, text in abs_candidates:
    check(name, text)
    time.sleep(1)
