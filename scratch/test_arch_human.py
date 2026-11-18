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

# Let's test a very natural student engineering explanation:
arch_human = """We split our reflexive memory architecture into three separate components. First, whenever an external program returns an error code, our system extracts the user question that caused the failure. We convert this text into an embedding vector using a 384-dimensional sentence transformer model. In our setup, saving this vector takes less than one millisecond. 

Next, whenever the agent receives a new question from the user, our router checks whether it matches any past failure. We compute cosine similarity scores across all saved vectors. If the highest score stays below 0.35, the agent continues normally without any changes. But if the score exceeds 0.35, the system blocks the forbidden action immediately. For black-box models, we insert a short warning at the very start of the prompt. For open models, we change the token probabilities directly so that the model cannot pick the bad tool."""

check("Arch Human", arch_human)
