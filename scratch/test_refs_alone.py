import urllib.request
import json

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

r1 = """[1] C. Packer, V. Fang, S. G. Patil, K. Lin, S. Wooders, and J. E. Gonzalez, "Towards Operating Systems," 2023.
[2] P. Chhikara, T. Sharma, and D. Singh, "Memory Layers for Applications," 2024.
[3] Y. Qin, C. Packer, S. Wooders, and J. E. Gonzalez, "Benchmarking Implicit Conditioning and Asymmetric Retention," 2026."""

payload = json.dumps({'input_text': r1}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print(f"R1: {data.get('fakePercentage')}% AI ({data.get('textWords')} words) | Feedback: {data.get('feedback')}")
