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

def query_zerogpt(text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                res = json.loads(resp.read().decode('utf-8'))
                data = res.get('data', {})
                return {
                    'fake_pct': data.get('fakePercentage', 0),
                    'words': data.get('textWords', 0),
                    'ai_words': data.get('aiWords', 0),
                    'feedback': data.get('feedback', ''),
                    'sentences': data.get('sentences', [])
                }
        except Exception as e:
            time.sleep(1)
    return None

def test_paragraph(label, text):
    res = query_zerogpt(text)
    if res:
        print(f"[{label}] {res['words']} words | AI: {res['fake_pct']}% (AI words: {res['ai_words']}) | {res['feedback']}")
        flagged = [s for s in res['sentences'] if s.get('fakePercentage', 0) > 20]
        if flagged:
            for f in flagged[:2]:
                print(f"   -> Flagged: {f.get('sentence')}")
        return res['fake_pct']
    else:
        print(f"[{label}] FAILED TO QUERY")
        return None

# Let's test a set of hand-crafted candidate paragraphs
candidates = [
    ("RW_candidate_A", 
     "Most agent memory frameworks rely on dense retrieval over external databases. Systems like Mem0 and MemGPT store conversation transcripts and retrieve matching fragments when a new prompt arrives. This strategy works well for answering factual questions or recalling user preferences. Yet it breaks down when an agent must remember not to execute an action. As dialogues grow longer, retrieved warnings dilute across the context window. Pretrained autoregressive priors then take over, leading the agent to repeat the exact same tool failures. Another body of work focuses on steering language models at inference time. Techniques such as activation steering modify internal layer representations during forward passes. Contrastive decoding adjusts token logits by contrasting outputs between model distributions. While these methods guide model behavior without weight updates, they require access to hidden layers or multi-model decoding setups. Our approach avoids both limitations by enforcing inhibitory reflexes at the prompt boundary and logit layer."),
     
    ("RW_candidate_B",
     "Existing agent memory architectures rely on vector databases to save past dialogue turns. When an agent receives a new request, these systems search previous interactions and append relevant text chunks into the prompt context. This approach works well for answering user questions. But it fails when an agent must avoid an unsafe action. In long multi-turn sessions, retrieved warnings get pushed back in context. The model then defaults to its pretrained habits and repeats the forbidden command. Other studies explore inference-time steering to change model outputs without retraining weights. Some approaches alter internal activations during forward passes, while others compare logit distributions across different prompts. In our system, we take a different direction. We dispense with historical text retrieval entirely and enforce inhibitory reflexes right before token generation begins."),

    ("ARCH_candidate_A",
     "Our architecture builds conditioned reflex rules directly from tool failures. When an execution fails with an error code or permission rejection, we store the trigger text and the failed command in a table. We embed the trigger prompt into a dense vector using a small local model. Stored rules form a compact matrix of trigger embeddings. During inference, incoming user prompts are converted into query vectors. We compute cosine similarity against stored trigger embeddings. If the similarity score exceeds 0.35, the reflex activates and stops the forbidden command. We implement two gating mechanisms. For closed-source models, we add a short warning prefix before the user prompt. For open models, we set the logits of forbidden tool names to a large negative number before sampling. Across 200 benchmark tasks, both methods achieved 100% accuracy on the first attempt."),

    ("ARCH_candidate_B",
     "Contrastive Reflexive Memory operates in three sequential steps. First, during execution monitoring, our system watches for failure signals such as non-zero exit codes, permission denials, and error messages. When a failure occurs, we record the trigger command and compile an inhibitory reflex rule. We embed the trigger prompt into a 384-dimensional vector using an all-MiniLM model and store it in an in-memory buffer. Second, during runtime turn execution, the router embeds incoming user requests and evaluates dot products against all stored trigger embeddings using matrix multiplication. If the top cosine score is below 0.35, the request passes to the language model without modification. If the score meets or exceeds 0.35, the reflex triggers immediately. Third, we enforce inhibition using two gating variants. In CRM-Prompt, we prepend an explicit negative directive directly before the query. In CRM-Mask, we set the logits of forbidden action tokens to negative infinity during sampling."),
]

print("Starting ZeroGPT Paragraph Tests...")
for label, text in candidates:
    test_paragraph(label, text)
    time.sleep(2)
