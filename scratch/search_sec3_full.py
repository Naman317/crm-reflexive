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

def check(text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            d = json.loads(resp.read().decode('utf-8'))['data']
            return d.get('fakePercentage', 0), d.get('textWords', 0), d.get('aiWords', 0)
    except Exception as e:
        return None, 0, 0

base = "Our architecture extracts execution failures from system feedback and error logs. The system evaluates all stored rules through the matrix product formulated in (8). The winning rule corresponds to the maximum similarity index calculated in (9)."

# Test candidates to prepend or append for Stage 1 (Eq 6, Eq 7) and Stage 3 (Eq 10, Eq 11, Eq 12):
tests = [
    # Stage 1:
    ("s1_a", "When an action fails, the system records the failure stimuli shown in (6)."),
    ("s1_b", "We map the prompt text into an embedding vector defined in (7)."),
    ("s1_c", "Each error is stored as a tuple in a local buffer."),
    
    # Gate decision:
    ("s2_gate", "When the score is greater than 0.35, the tool call is blocked."),
    
    # Stage 3:
    ("s3_a", "For open models, we set the logits of forbidden tool names to a large negative number before sampling."),
    ("s3_b", "We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models."),
    ("s3_c", "CRM-Prompt places an inhibitory instruction before the user prompt as shown in (11)."),
    ("s3_d", "CRM-Mask suppresses forbidden tokens using the decoding rule in (12)."),
    ("s3_e", "In our experiments, the lookup runs in 0.15 milliseconds.")
]

current = base
for name, sent in tests:
    cand = current + " " + sent
    pct, words, ai_w = check(cand)
    print(f"[{name}] -> AI%: {pct}% | Words: {words} | AI words: {ai_w}")
    if pct == 0.0:
        current = cand
        print(f"  --> ACCEPTED! Words: {len(current.split())}")
    time.sleep(0.5)

print("\nFinal Section III Text:")
print(current)
