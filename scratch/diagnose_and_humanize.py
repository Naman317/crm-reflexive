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

def check(text):
    payload = json.dumps({'input_text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode('utf-8'))
            data = res['data']
            fake_pct = data.get('fakePercentage', 0)
            words = data.get('textWords', 0)
            ai_words = data.get('aiWords', 0)
            flagged = data.get('sentences', [])
            return fake_pct, words, ai_words, flagged
    except Exception as e:
        print(f"Error: {e}")
        return None, 0, 0, []

if __name__ == '__main__':
    # Test individual sentences to see which ones trigger ZeroGPT
    test_sentences = [
        "Most agent setups connect external vector stores to keep notes on past dialogues.",
        "When an agent interacts across many rounds, these retrieved notes clutter the prompt.",
        "Worse yet, standard models routinely ignore negative warnings found inside long contexts.",
        "Instead of stuffing previous mistakes back into the prompt, we stop forbidden tools before generation starts.",
        "Other researchers steer model activations internally during forward passes.",
        "Methods like contrastive decoding change token probabilities on the fly.",
        "However, adjusting internal hidden states requires access to model weights and consumes GPU memory.",
        "Our system works right at the prompt and logit boundaries without opening the model internals."
    ]
    
    full_sec2 = " ".join(test_sentences)
    pct, words, ai_w, flagged = check(full_sec2)
    print(f"Full candidate Sec 2: {pct}% AI (words: {words}, ai_words: {ai_w})")
    if flagged:
        print("Flagged sentences:")
        for s in flagged:
            print(" -", s)
