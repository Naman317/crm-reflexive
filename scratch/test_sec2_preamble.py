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

# Candidate 1:
c1 = """In this section, we formulate the problem of negative-association blindness in language model agents. We first define our multi-turn evaluation setup and the mathematical equations governing action generation. Next, we introduce the First-Try Accuracy metric used to measure performance. Finally, we analyze why standard vector retrieval methods fail when conversations grow long."""

check("Candidate 1", c1)

# Candidate 2:
c2 = """To understand why software agents repeat mistakes, we need a precise mathematical framework. In this section, we outline the multi-turn interaction protocol used in our benchmark experiments. We then formalize the performance difference between positive preferences and negative inhibition. Lastly, we examine how distance in conversation context degrades traditional vector retrieval."""

check("Candidate 2", c2)

# Candidate 3:
c3 = """Before presenting our reflexive memory model, we define the interaction setting and formal metrics. We describe the sequential interaction steps between an agent and its software environment. We then quantify how baseline models fail on negative constraints compared to positive tasks. We also examine why standard retrieval databases fail to stop repeat errors across multi-turn dialogues."""

check("Candidate 3", c3)
