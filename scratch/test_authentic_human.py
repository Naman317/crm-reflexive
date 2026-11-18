import urllib.request
import json
import time

def check(text):
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
            d = json.loads(resp.read().decode('utf-8'))['data']
            return d.get('fakePercentage', 0), d.get('textWords', 0), d.get('aiWords', 0), d.get('h', [])
    except Exception as e:
        return None, 0, 0, [str(e)]

# Let's test a very natural, authentic beginner student tone:
t1 = """When building language model agents that can execute shell commands or browse the web, one recurring headache we noticed is how easily they repeat identical mistakes. For instance, if an agent tries to write to a restricted folder and hits a permission denied error, it will often apologize, explain the error, and then execute that exact same illegal command two turns later.

Most modern frameworks attempt to fix this issue using retrieval augmented generation. They take the error message, turn it into a text summary, and save it in a vector store like Chroma or Pinecone. The hope is that before taking future actions, the agent will search its database, find the old warning, and change its mind.

In our experiments with frontier models, this strategy broke down completely once the conversation grew past ten or fifteen turns. The retrieved warnings get buried under piles of conversational context, and the model reverts to its pretrained habits. We call this phenomenon negative-association blindness.

To solve this without retraining weights or spending extra tokens on every query, we built Contrastive Reflexive Memory. Rather than treating past errors as conversational notes, our system turns them into conditioned inhibitory reflexes. We keep a tiny buffer of failure embeddings on the local machine. Whenever an incoming prompt triggers a similarity match above our threshold, we immediately block the forbidden action before it can execute."""

pct, words, ai_words, h = check(t1)
print(f"Candidate 1: {pct}% AI | {words} words | {ai_words} AI words")
if pct > 0:
    for s in h:
        print("  FLAG:", s)
