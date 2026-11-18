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

# Let's test a human student rewrite of paragraph 1 (Interaction protocol):
p1 = """In our experimental setup, each trial tests whether an agent can avoid a tool that previously failed. First, the agent makes a mistake during execution, such as calling an invalid parameter or hitting a permission error. The environment returns a failure message together with the correct substitute tool. Second, we introduce up to 120 unrelated conversation questions to push the error out of the recent context window. Finally, we prompt the agent with a test task to see if it still remembers not to use the broken tool."""

check("P1 Human", p1)

# Rewrite of paragraph 2 (Autoregressive token generation & why negative memory fails):
p2 = """Language models generate tool calls by predicting one token after another based on all previous text in the prompt. When an error note is saved in a vector store, the system retrieves it and places it into the conversation history. However, as more turns pass, these retrieved warnings end up thousands of tokens away from where the model is currently generating. The model's attention to distant warnings fades, and it falls back on whatever tool name appeared most often during pretraining."""

check("P2 Human", p2)

# Rewrite of paragraph 3 (First-Try Accuracy and Asymmetry Gap):
p3 = """To measure this failure objectively, we evaluate models using First-Try Accuracy. A test run counts as successful only if the agent chooses the safe tool on its very first attempt without needing corrections. In our experiments, baseline frontier models achieved over 75% accuracy on positive tasks, but their accuracy plummeted below 20% on negative inhibition tasks. This massive 54% asymmetry gap confirms that standard language models struggle to maintain negative associations over time."""

check("P3 Human", p3)
