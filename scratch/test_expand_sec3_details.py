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

# Let's test an expanded, detailed Subsection III-A:
sub_a_cand1 = """During execution, an autonomous agent interacts with external software environments by running commands and scripts. When a command fails, the operating system returns an error code or an exception message. In our architecture, the failure extractor monitors these execution logs for runtime errors such as rejected permissions, syntax errors, or network deadlocks shown in (6). Instead of saving long conversation notes into a database, we save a short embedding of the user prompt. We use a local sentence transformer to convert the trigger prompt into a normalized embedding vector defined in (7). Each recorded error is stored as an inhibitory reflex tuple containing the trigger vector, the forbidden tool name, and the safe alternative."""

check("Sub A Expanded 1", sub_a_cand1)
