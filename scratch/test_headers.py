import urllib.request
import json

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

def check(name, text):
    payload = json.dumps({'input_text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        data = res['data']
        print(f"[{name}] Score: {data.get('fakePercentage')} % AI | Words: {data.get('textWords')} | Flagged: {data.get('sentences')}")

h1 = """Contrastive Reflexive Memory: Overcoming Negative-Association Blindness in Language Model Agents"""
h2 = """Naman, Member Two, Member Three, and Co-Author Advisor. Department of Computer Science and Engineering, Lovely Professional University, Phagwara, Punjab, India."""
h3 = """Abstract—Autonomous software agents execute actions by calling external tools in sequence."""

check("Title", h1)
check("Authors", h2)
check("Abstract prefix", h3)
