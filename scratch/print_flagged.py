import urllib.request
import json

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_craft_zerogpt.py', 'r', encoding='utf-8') as f:
    content = f.read()
craft = content.split('craft = """')[1].split('"""')[0]

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}
payload = json.dumps({'input_text': craft}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    for k, v in data.items():
        if k != 'originalParagraph':
            print(f"{k}: {v}")
