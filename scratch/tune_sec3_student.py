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

# Let's test student engineering voice for Section III:
# Stage 1:
s1_test = """When an agent runs a bad command, our failure extractor records the error feedback from the environment. The system listens for common runtime failures like syntax errors, permission problems, or deadlock warnings shown in (6). We convert the user prompt into a normalized embedding vector using (7) and save it in a local memory buffer."""

check("Stage 1 Student", s1_test)

# Stage 2:
s2_test = """During every subsequent turn, our semantic router compares incoming user inputs against all stored failure vectors. We perform a fast matrix multiplication using (8) to calculate similarity scores. The router selects the highest matching score using (9). When this score is higher than our safety threshold of 0.35, the decision gate in (10) triggers an inhibitory reflex."""

check("Stage 2 Student", s2_test)

# Stage 3:
s3_test = """Our framework provides two ways to stop forbidden actions during inference. For black-box models, CRM-Prompt adds a direct constraint right before the user prompt as formulated in (11). For open models, CRM-Mask sets the logits of prohibited action tokens to a large negative number using (12). This prevents the agent from selecting broken tools without requiring model retraining."""

check("Stage 3 Student", s3_test)
