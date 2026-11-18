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

p1 = "Language model agents exhibit strong capabilities in executing multistep tasks, but they struggle with basic negative conditioning when tools fail. In this paper, we introduce Contrastive Reflexive Memory, an architecture designed to capture error feedback, store forbidden actions in a fast vector table, and prevent repeat failures at test time."

p2 = "Modern software systems present diverse runtime hazards, including permission faults, rate limits, and network errors. When an agent triggers an error, it must update its execution strategy. Human engineers quickly adjust their workflow and stop invoking the broken call. Yet state of the art models like Claude 3.5, GPT-4o, and Qwen2.5 repeatedly attempt the forbidden action once several dialogue turns have elapsed."

p3 = "Vector stores and retrieval engines store previous turns as passive text notes. When context windows fill with dialogue, retrieved notes fail to suppress strong generation habits. Contrastive Reflexive Memory replaces retrieval with active reflex gating. We evaluate our implementation on ImplicitMemBench across 200 tasks. The lookup takes 0.15 milliseconds and achieves 100% accuracy on the first attempt."

detect("P1 alone", p1)
detect("P2 alone", p2)
detect("P3 alone", p3)
detect("P1 + P2 + P3", p1 + "\n\n" + p2 + "\n\n" + p3)
