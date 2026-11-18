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

# Section III Lead-in:
sec3_lead = """Our Contrastive Reflexive Memory framework operates across three distinct stages: failure extraction, semantic routing, and inhibitory actuation. Instead of searching long conversational logs, the system maintains a compact buffer of failure reflexes on the local machine."""

check("Sec 3 Lead", sec3_lead)

# Stage 1: Extractor
sec3_stage1 = """During runtime interaction, the failure extractor monitors tool execution logs for unconditioned error stimuli defined in (6). When an action fails, the system logs the trigger prompt and computes a normalized embedding vector using (7). The compiled reflex tuple is then added to the local reflex buffer."""

check("Sec 3 Stage 1", sec3_stage1)

# Stage 2: Semantic Router
sec3_stage2 = """At every new turn, the semantic router embeds the incoming user prompt and compares it against all stored rules using matrix multiplication in (8). The router identifies the best matching rule using the argmax index in (9). If the similarity exceeds our safety threshold, the gating function in (10) activates the inhibitory reflex."""

check("Sec 3 Stage 2", sec3_stage2)

# Stage 3: Actuator
sec3_stage3 = """For black-box models without logit access, CRM-Prompt inserts a direct contrastive instruction at offset zero as shown in (11). For open-weight models, CRM-Mask suppresses forbidden tool tokens directly at the decoding layer according to the logit masking rule in (12). Both methods ensure that the agent never repeats prohibited actions."""

check("Sec 3 Stage 3", sec3_stage3)
