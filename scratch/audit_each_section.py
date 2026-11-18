import urllib.request
import json
import re

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
            print(f"{name:30s} | {fake:5.1f}% AI | Words: {words:4d} | AI Words: {ai_words:4d} | Flagged: {len(h):2d}")
            return fake, h, text
    except Exception as e:
        print(f"{name:30s} | Error: {e}")
        return 100, [], text

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_full_ieee_structured.tex', 'r', encoding='utf-8') as f:
    tex = f.read()

# Extract sections
abstract_m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.DOTALL)
abstract = abstract_m.group(1).strip() if abstract_m else ""

# Helper to clean latex commands from a section
def clean_latex(s):
    # remove figures, tables, equations, inputs
    s = re.sub(r'\\begin\{(figure\*?|table\*?|equation\*?)\}.*?\\end\{\1\}', '', s, flags=re.DOTALL)
    s = re.sub(r'\\input\{.*?\}', '', s)
    s = re.sub(r'\\cite[pt]?\{.*?\}', '[1]', s)
    s = re.sub(r'\\[a-zA-Z]+(\[.*?\])?(\{.*?\})?', '', s)
    s = re.sub(r'[\$\{\}]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

sections = re.split(r'\\section\{([^}]+)\}', tex)
# sections[0] is preamble + abstract
# sections[1] is name 1, sections[2] is body 1, etc.

detect("Abstract (raw tex)", clean_latex(abstract))

for i in range(1, len(sections), 2):
    sec_name = sections[i]
    sec_body = sections[i+1]
    cleaned = clean_latex(sec_body)
    detect(f"Sec: {sec_name[:20]}", cleaned)
