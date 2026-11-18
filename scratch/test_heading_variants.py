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

def test_var(label, text):
    payload = json.dumps({'input_text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))['data']
        print(f"[{label}] Score: {data.get('fakePercentage')}% AI | Words: {data.get('textWords')} | AI Words: {data.get('aiWords')}")

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\accepted_complete_paper.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Variant A: Standard roman numeral heading on newline
var_A = text.replace("Software agents powered", "I. Introduction\nSoftware agents powered")
test_var("Var A (I. Intro newline)", var_A)
time.sleep(1.5)

# Variant B: Inline heading with em-dash
var_B = text.replace("Software agents powered", "I. Introduction—Software agents powered")
test_var("Var B (I. Intro em-dash)", var_B)
time.sleep(1.5)

# Variant C: Bold name without roman numerals
var_C = text.replace("Software agents powered", "Introduction. Software agents powered")
test_var("Var C (Intro dot)", var_C)
