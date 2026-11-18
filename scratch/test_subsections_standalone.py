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

# Let's test a distinct standalone paragraph for Subsection II-A:
sub_a = """In our test setup, an agent goes through three distinct stages: training, distraction, and testing. An agent makes an erroneous tool call and receives an error message from the environment. In the distraction stage, the user asks up to 120 unrelated questions. Afterwards, we evaluate whether the agent can successfully avoid the broken action on its first attempt. The language model generates its tool call by predicting next tokens sequentially according to the probability equation in (5)."""

check("Sub A Alone", sub_a)

# Now let's craft a standalone paragraph for Subsection II-B (Metrics & Paraphrase Decay):
sub_b = """We measure performance using the First-Try Accuracy metric defined in (4). When an agent encounters negative feedback, frontier models struggle to retain the constraint. Across our experiments, baseline models showed a 54% accuracy drop between positive preference tasks and negative inhibition tasks. When user prompts were rephrased, standard retrieval models failed to prevent repeat mistakes. Figure 2 shows the retention decay across different paraphrase tiers."""

check("Sub B Alone", sub_b)

# Now let's craft a standalone paragraph for Subsection II-C (Vector Store Limitations & Table I):
sub_c = """Most agent systems rely on vector databases to store past conversation notes. However, as dialogue history grows longer, attention over distant warning notes degrades. Table I summarizes how existing agent memory benchmarks handle long horizons and negative constraints. Instead of retrieving text notes into the prompt, our framework uses the cosine similarity score in (2) and applies contrastive prompt steering in (3)."""

check("Sub C Alone", sub_c)
