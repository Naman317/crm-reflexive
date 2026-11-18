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

# Let's test a richer text for Subsection III-A:
t_a = """Our architecture extracts execution failures from system feedback and error logs. Instead of saving long text notes into a database, we save a short embedding of the user prompt. When a tool fails, the system records the failure stimuli shown in (6). We map the prompt text into an embedding vector defined in (7). Our setup uses a small sentence transformer with 384 dimensions. Each stored reflex contains the trigger embedding, the forbidden action name, and the safe substitute tool."""

check("Test A Rich", t_a)

# Let's test a richer text for Subsection III-B:
t_b = """During interaction, the semantic router checks incoming prompts against all recorded failure cases. The system evaluates all stored rules through the matrix product formulated in (8). The winning rule corresponds to the maximum similarity index calculated in (9). When the score is greater than 0.35, the tool call is blocked. In our experiments, the lookup runs in 0.15 milliseconds on a commodity CPU."""

check("Test B Rich", t_b)

# Let's test a richer text for Subsection III-C:
t_c = """We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. CRM-Prompt places an inhibitory instruction before the user prompt as shown in (11). For open models, we set the logits of forbidden tool names to a large negative number before sampling using (12). This ensures that the agent avoids prohibited actions without needing parameter fine-tuning."""

check("Test C Rich", t_c)
