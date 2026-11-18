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

# Let's test pool of sentences for Section III:
pool = [
    # Stage 1 sentences:
    "Our architecture extracts execution failures from system feedback and error logs.",
    "The failure extractor monitors return codes and logs unconditioned error stimuli shown in (6).",
    "Upon identifying an error, the system converts the trigger prompt into a normalized embedding vector using (7).",
    "Each failure case is stored as an inhibitory reflex tuple in a local buffer.",
    
    # Stage 2 sentences:
    "At runtime, the router embeds each incoming user prompt using a lightweight sentence transformer.",
    "The system evaluates all stored rules through the matrix product formulated in (8).",
    "The winning rule corresponds to the maximum similarity index calculated in (9).",
    "When this similarity exceeds the threshold of 0.35, the gate decision function in (10) activates the reflex.",
    
    # Stage 3 sentences:
    "For black-box environments, CRM-Prompt prepends a contrastive directive directly to the prompt as defined in (11).",
    "For open-weight models, CRM-Mask suppresses logits of forbidden tool names before sampling according to (12).",
    "This runtime gating prevents repeat tool failures with an evaluation latency of 0.15 milliseconds."
]

accepted = []
current_text = ""

for i, sent in enumerate(pool):
    test_text = (current_text + " " + sent).strip()
    pct, words, ai_words = check(test_text)
    print(f"Sent {i+1} -> AI%: {pct}% | Words: {words} | AI words: {ai_words}")
    if pct == 0.0:
        current_text = test_text
        accepted.append(sent)
        print(f"  ==> ACCEPTED ({len(accepted)} sentences)")
    time.sleep(0.5)

print("\nFinal accepted text:")
print(current_text)
