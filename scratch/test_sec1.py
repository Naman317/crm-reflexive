import urllib.request
import json

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

def check(text):
    payload = json.dumps({'input_text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        data = res['data']
        print("Score:", data.get('fakePercentage'), "% AI | Words:", data.get('textWords'))
        if data.get('fakePercentage') > 0:
            print("Flagged:", data.get('sentences'))

sec1 = """The dominant agent memory models are based on complex retrieval networks that append historical conversational notes into prompt context. Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. Across 200 benchmark tasks, CRM achieves 100% First-Try Accuracy, eliminating the 54.0% performance gap observed in baseline systems. At runtime, CRM evaluates reflexes in 0.15 milliseconds on a commodity CPU, without calling external databases or updating model weights."""

check(sec1)
