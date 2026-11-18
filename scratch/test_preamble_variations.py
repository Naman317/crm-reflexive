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

# Let's test substantive engineering statements:
cand_a = """When an agent operates in real environments, actions produce error messages from operating systems and external APIs. A dependable agent needs to remember these errors so that it never repeats the forbidden action. Here, we define the mathematical framework for multi-turn tool interaction and examine why current models struggle with negative constraints."""

check("Cand A", cand_a)

cand_b = """Autonomous agents rely on sequential tool calls to complete user tasks. However, when an action fails, maintaining negative feedback across long dialogues remains a major challenge. We establish the formal interaction protocol, define our evaluation metrics, and analyze why standard retrieval memory degrades under conversational distraction."""

check("Cand B", cand_b)

cand_c = """Software agents learn positive tool preferences quickly, but they fail to maintain negative safety rules over extended dialogues. To analyze this problem systematically, we define the multi-turn interaction protocol and formalize the First-Try Accuracy metric across positive and negative conditioning tasks."""

check("Cand C", cand_c)
