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

def query_zgpt(text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode('utf-8'))['data']
                return {
                    'score': data.get('fakePercentage', 0),
                    'words': data.get('textWords', 0),
                    'ai_words': data.get('aiWords', 0),
                    'h': data.get('h', [])
                }
        except Exception as e:
            time.sleep(2)
    return None

# Let's test the title and authors:
header = """Reflexive Inhibition in LLM Agents: A Semantic Routing Approach to Negative-Association Memory
Ayush Raj Vishnu, Naman Sharma, Vishnu Narayanan P, Dr. Baljit Singh Saini
Department of Computer Science and Engineering, Lovely Professional University, Punjab, India
Reg. No: 12619994, Reg. No: 12610802, Reg. No: 12614204
ayushrajvishnu0311@gmail.com, namansharma3194@gmail.com, 222vishnunarayanan@gmail.com, baljit.22078@lpu.co.in"""

res = query_zgpt(header)
print("Header:", res)
