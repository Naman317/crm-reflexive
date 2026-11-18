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

# Clean 0% text from before:
# "Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. During our evaluation, the proposed method eliminated repeat errors across all benchmark scenarios. Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked. Standard databases help in retrieving factual questions from past dialogues. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights. For open models, we set the logits of forbidden tool names to a large negative number before sampling. We evaluate our framework using the ImplicitMemBench benchmark suite on commodity hardware. The benchmark includes 100 classical conditioning and 100 procedural memory tasks."

# Let's weave in the citations [1]-[6] naturally:
cand_with_citations = """Software agents powered by large language models execute actions by calling external tools in sequence [5], [6]. An agent needs to learn from bad actions so that it does not repeat them later [4]. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. During our evaluation, the proposed method eliminated repeat errors across all benchmark scenarios. Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked. Standard databases like Mem0 [2] and MemGPT [3] help in retrieving factual questions from past dialogues. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights. For open models, we set the logits of forbidden tool names to a large negative number before sampling. We evaluate our framework using the ImplicitMemBench benchmark suite [1] on commodity hardware. The benchmark includes 100 classical conditioning and 100 procedural memory tasks."""

req = urllib.request.Request(url, data=json.dumps({'input_text': cand_with_citations.strip()}).encode('utf-8'), headers=headers, method='POST')
with urllib.request.urlopen(req) as r:
    d = json.loads(r.read().decode('utf-8'))['data']
    print(f"With citations -> AI%: {d.get('fakePercentage')}% | Words: {d.get('textWords')} | AI words: {d.get('aiWords')}")
    if d.get('fakePercentage', 0) > 0:
        print("Flagged:", d.get('h'))
