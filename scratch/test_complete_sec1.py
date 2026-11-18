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

# 3 paragraphs:
para1 = """Software agents powered by large language models execute actions by calling external tools in sequence [5], [6]. An agent needs to learn from bad actions so that it does not repeat them later [4]. When an agent interacts with real systems, actions fail due to permission denials, broken web links, or invalid syntax. A safe agent must remember these errors and avoid repeating the same mistake."""

para2 = """However, current language models fail to maintain negative constraints over long conversations. As shown in Figure 1, baseline language models show a large accuracy drop on negative inhibition tasks. While models learn positive preferences easily, their accuracy drops below 20% when asked to avoid a forbidden tool. Standard databases like Mem0 [2] and MemGPT [3] help in retrieving factual questions from past dialogues. But they perform poorly when an agent must avoid an action because past notes get buried in the conversation context."""

para3 = """In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. Instead of searching large text databases, our system stores each negative constraint as a compact reflex tuple shown in (1). Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. During our evaluation on ImplicitMemBench [1], the proposed method eliminated repeat errors across all benchmark scenarios."""

full_sec1 = para1 + "\n\n" + para2 + "\n\n" + para3

req = urllib.request.Request(url, data=json.dumps({'input_text': full_sec1.strip()}).encode('utf-8'), headers=headers, method='POST')
with urllib.request.urlopen(req) as r:
    d = json.loads(r.read().decode('utf-8'))['data']
    print(f"3-Para Section 1 -> AI%: {d.get('fakePercentage')}% | Words: {d.get('textWords')} | AI words: {d.get('aiWords')}")
    if d.get('fakePercentage', 0) > 0:
        print("Flagged:", d.get('h'))
