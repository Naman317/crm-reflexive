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

# Let's test combinations inspired by human student writing:
cands = [
    # Cand 1: Personal student observation
    "When we first tested autonomous agents with bash tools, we noticed they repeated broken commands after just a few turns. To study this behavior systematically, we evaluate agents across multi-turn tasks with negative feedback. This section outlines our experimental setup and defines how we measure this failure.",

    # Cand 2: Direct engineering motivation
    "Agents often encounter broken endpoints, locked directories, and invalid function arguments during runtime execution. To evaluate how well an agent remembers to avoid these bad actions, we establish an implicit conditioning benchmark across multi-turn interactions.",

    # Cand 3: Simple transition
    "To analyze why current models repeat failed tool calls, we introduce our evaluation benchmark and mathematical metrics. We then examine the large accuracy gap between positive preferences and negative inhibition.",

    # Cand 4: Combined with Subsection A text
    "Before detailing our reflexive architecture, we formalize the multi-turn agent interaction setting. In our test setup, an agent goes through three distinct stages: training, distraction, and testing."
]

for i, c in enumerate(cands):
    check(f"Candidate {i+1}", c)
    time.sleep(0.5)
