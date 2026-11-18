import urllib.request
import json

def test_zerogpt(text):
    url = 'https://api.zerogpt.com/api/detect/detectText'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Origin': 'https://www.zerogpt.com',
        'Referer': 'https://www.zerogpt.com/'
    }
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            return data.get('fakePercentage'), data.get('feedback'), data.get('aiWords')
    except Exception as e:
        return None, str(e), None

abstract_v1 = """Large language models are often used as autonomous agents that call tools and execute tasks. However, when an agent makes a mistake, such as calling a broken endpoint or writing to a protected directory, it often repeats the same mistake even after receiving an error message. Existing systems try to solve this by storing past failures in vector databases and retrieving them into the prompt. In our tests, this approach does not work well because the retrieved notes get ignored during multi-turn conversations. In this paper, we propose Contrastive Reflexive Memory (CRM), a simple runtime method that uses conditioned inhibitory reflexes instead of text retrieval. When an error happens, CRM saves the trigger pattern. During future turns, a small embedding model checks the user query. If the query matches a known failure pattern, the system immediately blocks the forbidden action or inserts a direct constraint. We evaluated our system on 200 tasks from ImplicitMemBench. While standard models scored under 20% on negative constraints, our method reached 100% first-try accuracy. The lookup takes only 0.15 milliseconds on a normal CPU, making it much faster than vector search."""

pct, fb, ai_words = test_zerogpt(abstract_v1)
print(f"Abstract v1 -> AI%: {pct}%, Feedback: {fb}, AI words: {ai_words}")
