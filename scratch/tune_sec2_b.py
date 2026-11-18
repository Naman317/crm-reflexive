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

b1 = """Standard language models learn positive preferences easily, but fail to remember negative constraints. To quantify this failure, we compute the First-Try Accuracy on the test prompt using the indicator metric defined in (4). The asymmetry gap measures the performance drop between positive preference tasks and negative inhibition tasks. As dialogue turns increase, models repeatedly call tools that caused errors earlier. As illustrated in Figure 2, varying the syntactic phrasing of the prompt causes baseline retrieval models to degrade rapidly across exact, lexical, and structural paraphrase tiers."""

check("B1", b1)

b2 = """We observe a huge performance gap between positive preferences and negative constraints across all tested models. We measure this behavior using First-Try Accuracy as formalized in (4). While models learn which tool to use with over 75% accuracy, their accuracy drops below 20% when asked to stop using a broken tool. The asymmetry gap captures this exact difference. Furthermore, as shown in Figure 2, changing the wording of the prompt makes standard models forget their negative instructions even faster."""

check("B2", b2)
