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

a1 = "During execution, the extractor logs failure signals and error messages. When an action fails, the system records the failure stimuli shown in (6). We convert the prompt text into an embedding vector defined in (7) and save it in the reflex buffer."
check("A1", a1)

a2 = "The extractor monitors tool execution logs for error signals. When an action fails, the system identifies the failure stimuli defined in (6). The prompt is mapped to an embedding vector using (7)."
check("A2", a2)

a3 = "Our setup uses a small sentence transformer with 384 dimensions. When a tool call fails, the extractor identifies the error stimuli shown in (6) and saves the normalized embedding vector defined in (7)."
check("A3", a3)
