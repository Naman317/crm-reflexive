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

# Draft for Subsection A:
sec2_a = """An implicit memory episode contains three sequential phases: learning, interference, and evaluation. During learning, the environment returns an execution error code when the agent invokes a forbidden action. The interference phase introduces up to 120 unrelated dialogue turns. Finally, the test prompt evaluates whether the agent remembers to avoid the broken tool. At each turn t, the model receives an input prompt x_t and generates a sequence of tool tokens according to the autoregressive probability distribution shown in (5)."""

check("Sec 2A", sec2_a)

# Draft for Subsection B:
sec2_b = """Standard language models learn positive preferences easily, but fail to remember negative constraints. To quantify this failure, we compute the First-Try Accuracy on the test prompt using the indicator metric defined in (4). The asymmetry gap measures the performance drop between positive preference tasks and negative inhibition tasks. As dialogue turns increase, models repeatedly call tools that caused errors earlier. Furthermore, as illustrated in Fig. 2, varying the syntactic phrasing of the prompt causes baseline retrieval models to degrade rapidly across exact, lexical, and structural paraphrase tiers."""

check("Sec 2B", sec2_b)

# Draft for Subsection C:
sec2_c = """Most agent frameworks store past errors as declarative sentences in a vector database like Mem0 or MemGPT. However, as conversation turns accumulate, attention over distant retrieval context decreases. In contrast, our system evaluates query similarity directly using the cosine similarity score defined in (2). When the score exceeds the safety threshold, the system injects a contrastive directive into the prompt as defined in (3). Table I summarizes the differences between passive context retrieval and our reflexive inhibition mechanism."""

check("Sec 2C", sec2_c)
