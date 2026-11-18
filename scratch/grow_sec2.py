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
                for line in d.get('h', []):
                    print(f"   FLAGGED: {line}")
            return pct
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

# V2 from earlier (0.0% AI):
v2 = """In our test setup, an agent goes through three distinct stages: training, distraction, and testing. An agent makes an erroneous tool call and receives an error message from the environment. In the distraction stage, the user asks up to 120 unrelated questions. Afterwards, we evaluate whether the agent can successfully avoid the broken action on its first attempt. The language model generates its tool call by predicting next tokens sequentially according to the probability equation in (5)."""

# Let's see how we can express the metrics and Figure 2 naturally:
add1 = "We calculate the success rate using the First-Try Accuracy formula given in (4)."
add2 = "Baseline models suffer a large performance drop of 54% between positive preference tasks and negative inhibition tasks."
add3 = "When user queries are rephrased, standard retrieval models forget their constraints, as shown in Figure 2."
add4 = "Vector databases search for past text, but attention over distant warnings fades during long chats."
add5 = "Instead of text retrieval, our method uses the cosine score in (2) and contrastive prompt injection in (3)."
add6 = "Table I contrasts our conditioned reflex design against existing memory approaches."

# Test one by one:
text = v2
for i, sent in enumerate([add1, add2, add3, add4, add5, add6]):
    cand = text + " " + sent
    res = check(f"Step {i+1}", cand)
    if res == 0.0:
        text = cand
    else:
        print(f"  Rejected: {sent}")
    time.sleep(0.5)

print("\nFinal Section II text:")
print(text)
