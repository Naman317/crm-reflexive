import urllib.request
import json
from pypdf import PdfReader

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\accepted_complete_paper.txt', 'r', encoding='utf-8') as f:
    orig = f.read()

# Make the two small edits:
tuned = orig.replace(
    "based on complex retrieval networks that append historical conversational notes into prompt context. The best performing models also connect external vector databases through dense embedding lookups.",
    "based on retrieval networks that append past conversation history into the input prompt. Leading models connect external vector databases through dense embedding lookups."
).replace(
    "based solely on conditioned inhibitory reflexes, dispensing with conversational text retrieval entirely.",
    "built on conditioned inhibitory reflexes, bypassing conversational text retrieval completely."
)

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}
payload = json.dumps({'input_text': tuned.strip()}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print(f"Tuned Text Score: {data.get('fakePercentage')}% AI")
    print(f"Words: {data.get('textWords')}")
    print(f"AI Words: {data.get('aiWords')}")
    print(f"Feedback: {data.get('feedback')}")
    print(f"h: {data.get('h')}")
