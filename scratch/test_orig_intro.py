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

def detect(text):
    payload = json.dumps({'input_text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            return data
    except Exception as e:
        return {'error': str(e)}

orig_intro = """Deploying large language models as autonomous agents requires them to internalize feedback and avoid repeating penalized operations. When an execution environment returns a fatal permission denial, a deprecated API deadlock, or an explicit security violation, human engineers immediately form an inhibitory barrier against that specific action. Frontier models, conversely, struggle with negative associations.

Recent systematic evaluations on ImplicitMemBench expose this vulnerability. While models such as Qwen2.5-7B, Claude-4.1-Opus, and GPT-5 achieve 74% to 85% First-Try Accuracy when learning positive user formatting preferences, their performance degrades to 17.6% to 38.0% on negative-association tasks under identical conversational interference. Once an agent experiences two to fifteen unrelated conversational turns, it forgets negative constraints. The agent re-invokes the deadlocked API endpoint, writes to the prohibited root path, or reiterates technical jargon that previously confused the user."""

res = detect(orig_intro)
print(f"Original Intro: {res.get('fakePercentage')}% AI ({res.get('textWords')} words) | Feedback: {res.get('feedback')}")
