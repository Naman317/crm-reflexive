import urllib.request
import json

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

sec2_candidate = "Vector stores work well for answering questions about previous conversations. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights. We block prohibited actions before the model begins decoding tokens. We place our safety rules at the input prompt or mask forbidden tokens during decoding."

payload = json.dumps({'input_text': sec2_candidate}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print(f"Sec 2 cand: {data.get('fakePercentage')}% AI ({data.get('textWords')} words) | Feedback: {data.get('feedback')}")
