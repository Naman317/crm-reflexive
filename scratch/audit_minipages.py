from pypdf import PdfReader
import urllib.request
import json

reader = PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_minipages_author.pdf')
txt = "\n\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

print("Extracted text:")
print(txt[:400])

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

payload = json.dumps({'input_text': txt}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print("\n================== ZEROGPT SCORE ==================")
    print("AI Percentage:", data.get('fakePercentage'), "%")
    print("Feedback:", data.get('feedback'))
    print("Words:", data.get('textWords'))
    print("AI Words:", data.get('aiWords'))
    print("Flagged:", data.get('sentences'))
