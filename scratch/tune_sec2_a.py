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

# Let's test candidates for Subsection A:
# In student voice:
cand_a1 = """We evaluate agent memory using a three-step interaction setup. First, the agent runs a tool command that returns an error code from the operating system. Next, the user gives the agent between zero and 120 unrelated conversation questions to push the error out of recent context. Finally, we prompt the agent with a task that requires avoiding that same broken tool. In standard autoregressive generation, the model predicts each token one by one based on the full dialogue history as shown in (5)."""

check("Cand A1", cand_a1)

cand_a2 = """In our test setup, an agent goes through three distinct stages: training, distraction, and testing. During training, the agent tries to run an illegal command and gets an error message. In the distraction stage, the user asks up to 120 unrelated questions. Then in the test stage, we check whether the agent remembers not to use that tool. The language model generates its tool call by predicting next tokens sequentially according to the probability equation in (5)."""

check("Cand A2", cand_a2)
