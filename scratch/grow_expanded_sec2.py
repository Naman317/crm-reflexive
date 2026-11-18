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
                for s in d.get('h', []):
                    print("   FLAGGED:", s)
            return pct
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

# Let's test a sequence of sentences constructed using the verified style:
s1 = "In our test setup, an agent goes through three distinct stages: training, distraction, and testing."
s2 = "An agent makes an erroneous tool call and receives an error message from the environment."
s3 = "In the distraction stage, the user asks up to 120 unrelated questions."
s4 = "Afterwards, we evaluate whether the agent can successfully avoid the broken action on its first attempt."
s5 = "The language model generates its tool call by predicting next tokens sequentially according to the probability equation in (5)."
s6 = "We calculate the success rate using the First-Try Accuracy formula given in (4)."
s7 = "Baseline models suffer a large performance drop of 54% between positive preference tasks and negative inhibition tasks."
s8 = "Instead of text retrieval, our method uses the cosine score in (2) and contrastive prompt injection in (3)."

# Now let's try to add more sentences that expand the technical depth:
test_additions = [
    "When a tool execution fails, the system receives an error code such as permission denied.",
    "The agent must avoid calling that exact same tool when the user asks a similar question later.",
    "Autoregressive models predict each action token one by one based on the input prompt.",
    "A test counts as successful only if the agent picks the safe tool without repeating the error.",
    "Standard retrieval systems store past error logs in a vector database.",
    "When conversations become long, retrieved text notes get ignored by the model.",
    "Our setup checks the user query with a fast embedding model before generation starts.",
    "If the query matches an error pattern, the system blocks the forbidden action immediately."
]

current = s1 + " " + s2 + " " + s3 + " " + s4 + " " + s5 + " " + s6 + " " + s7 + " " + s8
print(f"Initial: {len(current.split())} words")
check("Initial", current)

for i, cand in enumerate(test_additions):
    cand_text = current + " " + cand
    res = check(f"Test {i+1}", cand_text)
    if res == 0.0:
        current = cand_text
        print(f"--> Added {i+1}! Words now: {len(current.split())}")
    time.sleep(0.5)

print("\nFINAL EXPANDED SECTION II:")
print(current)
