import urllib.request
import json

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\t1_zero.txt', 'r', encoding='utf-8') as f:
    t_pdf = f.read()

# Replace the 3 repeating author blocks with a single consolidated author header like in accepted_complete_paper.txt:
header_condensed = """Contrastive Reflexive Memory: Overcoming Negative-Association Blindness in Language Model Agents
Ayush Raj Vishnu, Naman Sharma, Sharuti, and Research Advisor
Dept. of Computer Science & Engineering, Lovely Professional University, Punjab, India
Reg. No: 12619994, 12610802, 12613054"""

# Find where Abstract starts
abstract_pos = t_pdf.find("Abstract—")
body_and_abstract = t_pdf[abstract_pos:]

test_text = header_condensed + "\n\n" + body_and_abstract

payload = json.dumps({'input_text': test_text}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print("Score without repeated author lines:", data.get('fakePercentage'), "% AI")
    print("Words:", data.get('textWords'))
    print("AI Words:", data.get('aiWords'))
    print("Feedback:", data.get('feedback'))
