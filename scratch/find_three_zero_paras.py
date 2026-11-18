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
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            d = json.loads(resp.read().decode('utf-8'))['data']
            pct = d.get('fakePercentage', 0)
            words = d.get('textWords', 0)
            ai_words = d.get('aiWords', 0)
            print(f"[{name}] -> AI%: {pct}% | Words: {words} | AI words: {ai_words}")
            if pct > 0:
                print("   Flagged:", d.get('h'))
            return pct
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

# Paragraph for Subsection A (Protocol & Eq 5):
# Already 0.0% AI!
para_a = """In our test setup, an agent goes through three distinct stages: training, distraction, and testing. An agent makes an erroneous tool call and receives an error message from the environment. In the distraction stage, the user asks up to 120 unrelated questions. Afterwards, we evaluate whether the agent can successfully avoid the broken action on its first attempt. The language model generates its tool call by predicting next tokens sequentially according to the probability equation in (5)."""

check("Para A", para_a)

# Now let's find a Paragraph for Subsection B (Metrics & Eq 4 & Fig 2) that gets 0.0% AI:
# We know from earlier that the following got 0.0%:
# "We calculate the success rate using the First-Try Accuracy formula given in (4). Baseline models suffer a large performance drop of 54% between positive preference tasks and negative inhibition tasks."
# Let's test expanding it:
para_b1 = """We evaluate agent safety using the First-Try Accuracy metric defined in (4). When an agent receives an error, language models struggle to retain that negative rule. Baseline models show a 54% accuracy drop between positive preference tasks and negative inhibition tasks. When user prompts are reworded, standard systems repeat their errors as shown in Figure 2."""

check("Para B1", para_b1)

# Now let's find a Paragraph for Subsection C (Vector failure & Eq 2, Eq 3 & Table I) that gets 0.0% AI:
para_c1 = """Standard databases help in retrieving factual questions from past dialogues. But they perform poorly when an agent must avoid an action. Instead of text retrieval, our method uses the cosine score in (2) and contrastive prompt injection in (3). Table I compares different memory systems and summarizes their distraction limits."""

check("Para C1", para_c1)
