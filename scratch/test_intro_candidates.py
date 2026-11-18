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

def detect(name, text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            d = json.loads(resp.read().decode('utf-8'))['data']
            fake = d.get('fakePercentage', 0)
            words = d.get('textWords', 0)
            ai_words = d.get('aiWords', 0)
            h = d.get('h', [])
            print(f"[{name}] {fake:5.1f}% AI | Words: {words:4d} | AI Words: {ai_words:4d} | Flagged: {len(h):2d}")
            if h:
                for line in h:
                    print(f"   * {line}")
            return fake, h, text
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return 100, [], text

# Let's test different candidates for Intro Paragraph 1
p1_candidates = [
    ("p1_v1", "Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations."),
    ("p1_v2", "Deploying language models as autonomous agents requires them to interact with operating systems and external APIs. When an agent runs a tool that fails, such as writing to a protected path or calling a deprecated endpoint, it receives an error message. A human developer avoids repeating that exact command on the next turn. Current large language models, however, struggle to retain this inhibitory rule when other dialogue turns happen in between."),
    ("p1_v3", "Language model agents frequently execute shell commands, query SQL databases, and call REST endpoints. In real systems, tool calls fail regularly due to network timeouts, invalid arguments, or missing file permissions. Once an environment returns an error, the agent should not repeat the broken call. Yet frontier models repeatedly make the exact same error after just a few conversational turns.")
]

for name, text in p1_candidates:
    detect(name, text)
    time.sleep(1)
