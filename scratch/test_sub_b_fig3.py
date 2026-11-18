import urllib.request
import json

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

def check(name, text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            d = json.loads(resp.read().decode('utf-8'))['data']
            pct = d.get('fakePercentage', 0)
            words = d.get('textWords', 0)
            ai_words = d.get('aiWords', 0)
            print(f"[{name}] -> AI%: {pct}% | Words: {words} | AI words: {ai_words}")
            if pct > 0:
                print("   Flagged:", d.get('h'))
            return pct
    except Exception as e:
        print(f"[{name}] Error: {e}")
        return None

t1 = """The router embeds user queries using transformer representations [8]. The system evaluates all stored rules through the matrix product formulated in (8). The winning rule corresponds to the maximum similarity index calculated in (9). When the score exceeds 0.35, the tool call is blocked as shown in (10). We plot the threshold sweep across different values in Fig. 3."""
check("T1", t1)

t2 = """The router embeds incoming queries using transformer representations [8]. The system evaluates stored rules using the matrix product in (8). The winning rule corresponds to the maximum similarity index in (9). When the cosine similarity exceeds 0.35, the gate triggers according to (10). As shown in Fig. 3, sweeping tau from 0.15 to 0.65 balances precision and recall."""
check("T2", t2)

t3 = """The router embeds user queries using standard transformer features [8]. The system compares the query against stored vectors using the matrix product in (8). Equation (9) identifies the best match index. When the similarity score exceeds 0.35 in (10), the gate blocks the forbidden tool. In Fig. 3, we show the empirical threshold trade-off across different tau values."""
check("T3", t3)

t4 = """We route incoming prompts using transformer embeddings [8]. The system scans stored rules using the matrix operation in (8). We select the maximum scoring rule according to (9). The decision gate in (10) activates whenever the score exceeds 0.35. Fig. 3 illustrates the empirical trade-off when varying the threshold parameter."""
check("T4", t4)
