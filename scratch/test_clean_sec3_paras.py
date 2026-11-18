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

# Paragraph 1 (Lead-in & Stage 1):
p1 = """Our architecture extracts execution failures from system feedback and error logs. Instead of saving long text notes into a database, we save a short embedding of the user prompt. When a tool fails, the system records the failure stimuli shown in (6). We map the prompt text into an embedding vector defined in (7)."""

check("P1", p1)

# Paragraph 2 (Stage 2 Router):
p2 = """The system evaluates all stored rules through the matrix product formulated in (8). The winning rule corresponds to the maximum similarity index calculated in (9). When the score is greater than 0.35, the tool call is blocked."""

check("P2", p2)

# Paragraph 3 (Stage 3 Actuator):
p3 = """We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. CRM-Prompt places an inhibitory instruction before the user prompt as shown in (11). For open models, we set the logits of forbidden tool names to a large negative number before sampling using (12)."""

check("P3", p3)
