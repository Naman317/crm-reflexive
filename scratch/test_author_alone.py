import urllib.request
import json

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

author_only = """Ayush Raj Vishnu
Dept. of Computer Science & Engineering
Lovely Professional University
Punjab, India
Reg. No: 12619994
ayushrajvishnu0311@gmail.com

Naman Sharma
Dept. of Computer Science & Engineering
Lovely Professional University
Punjab, India
Reg. No: 12610802
namansharma3194@gmail.com

Sharuti
Dept. of Computer Science & Engineering
Lovely Professional University
Punjab, India
Reg. No: 12613054
shrutirana11122@gmail.com

Co-Author Advisor
Dept. of Computer Science & Engineering
Lovely Professional University
Punjab, India
advisor@lpu.co.in"""

payload = json.dumps({'input_text': author_only}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print("Author block alone:", data.get('fakePercentage'), "% AI")
    print("Words:", data.get('textWords'))
