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

# Sub A with classical conditioning citations [9], [10]:
sub_a_cit = """Drawing inspiration from conditioned inhibition in classical psychology [9], [10], our architecture extracts execution failures from system feedback and error logs. Instead of saving long text notes into a database, we save a short embedding of the user prompt. When a tool fails, the system records the failure stimuli shown in (6). We map the prompt text into an embedding vector defined in (7)."""

check("Sub A Cit", sub_a_cit)

# Sub B with transformer citation [8] and Figure 3:
sub_b_cit = """The router embeds user queries using transformer representations [8]. The system evaluates all stored rules through the matrix product formulated in (8). The winning rule corresponds to the maximum similarity index calculated in (9). When the score is greater than 0.35, the tool call is blocked. Figure 3 shows the threshold calibration curve sweeping tau from 0.15 to 0.65, demonstrating optimal recall without false alarms."""

check("Sub B Cit", sub_b_cit)

# Sub C with in-context steering [11], DPO [7], and open models [12]:
sub_c_cit = """We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models like LLaMA [12]. CRM-Prompt places an inhibitory instruction before the user prompt as shown in (11), leveraging in-context conditioning [11] without model retraining [7]. For open models, we set the logits of forbidden tool names to a large negative number before sampling using (12)."""

check("Sub C Cit", sub_c_cit)
