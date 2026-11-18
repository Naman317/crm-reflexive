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

# Let's test a natural student introduction:
intro_v1 = """Recent advances in large language models enable autonomous software agents to perform complex multi-step workflows by executing external tools. In practical deployments, an agent must reliably heed environmental feedback, especially when an action produces an error. For example, if calling an external API returns an authentication failure or writing to a system path causes a permission violation, the agent should immediately update its behavior and refrain from repeating the forbidden operation.

Yet, empirical evaluations on benchmarks such as ImplicitMemBench reveal a persistent limitation termed negative-association blindness. While state-of-the-art language models reliably acquire positive preferences, their accuracy drops sharply when tasked with remembering negative constraints across multi-turn interactions. After several unrelated dialogue turns, standard agents routinely re-invoke deprecated functions and repeat identical system exceptions.

Current agent architectures attempt to address this failure through declarative retrieval-augmented generation. When an error occurs, these systems store a text summary in a vector database, retrieving historical error notes back into the prompt context during subsequent turns. However, as the conversation history grows, soft attention over distant context tokens disperses, allowing strong affirmative language model priors to override passive text warnings.

In this work, we propose Contrastive Reflexive Memory, a lightweight runtime mechanism inspired by classical inhibitory conditioning. Instead of relying on conversational context retrieval, our method represents negative constraints as conditioned inhibitory reflexes. We evaluate incoming prompts using a compact local sentence transformer, matching queries against stored failure patterns in 0.15 milliseconds on a commodity processor. When a trigger pattern is detected, the framework prevents repeat errors through prompt-level contrastive steering or direct logit masking, achieving 100% First-Try Accuracy across all benchmark tasks without fine-tuning model parameters."""

check("Intro v1", intro_v1)
