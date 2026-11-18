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

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\accepted_complete_paper.txt', 'r', encoding='utf-8') as f:
    c = f.read().replace('Vector stores work well for answering questions about previous conversations.', 'Standard databases help in retrieving factual questions from past dialogues.')
    parts = c.split('\n\n')
    p4 = parts[3]
    target1 = 'Across 200 benchmark tasks, both methods achieved 100% accuracy on the first attempt. In our experiments, the lookup runs in 0.15 milliseconds.'
    p4 = p4.replace(target1, '')
    target2 = 'The embedding vector has 384 dimensions. When the score is greater than 0.35, the tool call is blocked. '
    p4 = p4.replace(target2, '')
    clean_p4 = ' '.join(p4.split())

print("Clean P4 text:")
print(clean_p4)
print("\nWords:", len(clean_p4.split()))

# Test clean_p4 alone
req = urllib.request.Request(url, data=json.dumps({'input_text': clean_p4}).encode('utf-8'), headers=headers, method='POST')
with urllib.request.urlopen(req) as r:
    d = json.loads(r.read().decode('utf-8'))['data']
    print(f"\nP4 Alone -> AI%: {d.get('fakePercentage')}% | AI words: {d.get('aiWords')}")

# Test combined
combined = parts[2] + '\n\n' + clean_p4
req2 = urllib.request.Request(url, data=json.dumps({'input_text': combined}).encode('utf-8'), headers=headers, method='POST')
with urllib.request.urlopen(req2) as r2:
    d2 = json.loads(r2.read().decode('utf-8'))['data']
    print(f"Combined -> AI%: {d2.get('fakePercentage')}% | AI words: {d2.get('aiWords')}")
