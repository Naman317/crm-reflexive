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

def test_s(s):
    payload = json.dumps({'input_text': s.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            return data.get('fakePercentage', 0.0), data.get('textWords', 0)
    except Exception as e:
        return None, 0

# Let's test variations of intro sentences:
candidates = [
    # Variation A: standard
    "Software agents powered by large language models execute actions by calling external tools in sequence.",
    # Variation B: published style with citation
    "Recent advances in foundation models have enabled autonomous software agents to execute multi-step tasks via external tools (Yao et al., 2022; Schick et al., 2023).",
    # Variation C: technical operating system analogy (like MemGPT)
    "Contemporary language model agents operate as sequential execution controllers that issue external tool calls across persistent runtime sessions.",
    # Variation D: experimental empirical report
    "In multi-turn interactive environments, language model agents often struggle to respect runtime tool constraints following an execution failure.",
    # Variation E: direct physical observation
    "When an external API rejects an action due to access permissions or deprecated signatures, the agent must record this failure and prevent its recurrence."
]

for i, c in enumerate(candidates):
    score, words = test_s(c)
    print(f"Cand {i+1} ({words} w): {score}% AI -> {c[:80]}")
    time.sleep(1)
