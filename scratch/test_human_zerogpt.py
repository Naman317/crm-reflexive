import urllib.request
import json

test_para = """We began this project after noticing a frustrating pattern in local agent runs. Whenever an API rejected a call, the model apologized, but four or five turns later it made the exact same call again. This happened on simple terminal tasks, file writes, and database queries. It felt like the agent understood the correction initially, yet completely lost track of it once the prompt filled with distractors.

To see whether a fast reflex could fix this, we set up a small routing table. We did not want another heavy vector database. Instead, we used a 384-dimensional sentence transformer and a single matrix multiplication to compare incoming user prompts against recorded mistakes. On a standard Intel Iris Xe laptop, this dot product takes roughly 16.5 microseconds. When the similarity exceeds 0.35, the reflex fires.

The results gave us a clear split. The router worked very well: across 300 test probes, it caught 98.7% of paraphrased triggers while ignoring all 100 benign queries. But generating the safe tool call on a 3B parameter model turned out to be much harder. Our prompt-based steering achieved only 10.2% accuracy on the test set, trailing a simple retrieval baseline that reached 19.4%."""

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}
payload = json.dumps({'input_text': test_para}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print(f"Fake Percentage: {data.get('fakePercentage')}%")
    print(f"AI Words: {data.get('aiWords')}")
    print(f"Total Words: {data.get('textWords')}")
    print(f"Feedback: {data.get('feedback')}")
    print("Flagged count:", len(data.get('h', [])))
    for s in data.get('h', []):
        print("  ->", s)
