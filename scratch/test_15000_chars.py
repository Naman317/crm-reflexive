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

def audit_text(text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            print(f"Characters: {len(text)} | Words: {data.get('textWords')} | Score: {data.get('fakePercentage')}% | AI Words: {data.get('aiWords')}")
            print(f"Feedback: {data.get('feedback')}")
            if data.get('fakePercentage', 0) > 0:
                print(f"Flagged sentences count: {len(data.get('h', []))}")
                for h in data.get('h', [])[:5]:
                    print(f"   FLAGGED: {h}")
            return data.get('fakePercentage')
    except Exception as e:
        print("Error:", e)
        return None

# Let's test the text from scratch/accepted_complete_paper.txt:
with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\accepted_complete_paper.txt', 'r', encoding='utf-8') as f:
    acc_text = f.read()

print("Testing accepted_complete_paper.txt:")
audit_text(acc_text)
