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

# Paragraph A: Formal interaction and error handling
para_a = """During each turn, an autonomous agent reads a user command and decides which tool to run. If an action fails because of a file lock, network error, or missing permission, the environment returns an error traceback. In a safe system, the agent should update its internal state so that it does not repeat the exact same command. Standard training teaches language models how to write code and call functions, but it does not give them a reliable way to remember negative constraints across extended conversations."""

check("Para A", para_a)

# Paragraph B: Multi-turn distraction and prompt dilution
para_b = """To test how well models remember constraints over time, we insert unrelated conversation turns between the initial error and the evaluation prompt. In our experiments, we increased this distraction gap from zero turns all the way to 120 turns. When the conversation history grows large, retrieved error notes get pushed far back in the prompt context. Under standard attention mechanisms, the model pays less attention to distant warnings and defaults back to its pretrained habits."""

check("Para B", para_b)

# Paragraph C: Asymmetry gap and paraphrase analysis
para_c = """This leads to a severe performance difference between positive and negative tasks. Across our evaluations, frontier models scored around 75% when asked to follow a positive preference, but dropped below 20% on negative inhibition tasks. When user prompts were rephrased using synonyms or inverted syntax, standard retrieval systems forgot their constraints even faster. Our reflexive approach resolves this issue by routing queries through a dedicated trigger buffer, avoiding text retrieval entirely."""

check("Para C", para_c)
