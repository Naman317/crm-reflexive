import urllib.request
import json

# Text from the user's prompt (rebuilt paper text)
user_paper_text = """Abstract—Language agents that call tools tend to repeat actions that have already failed. The problem is sharpest when the lesson is a prohibition (“do not call this endpoint”) rather than a preference, and published benchmark results suggest that frontier models handle prohibitions far worse than preferences. We describe Contrastive Reflexive Memory (CRM), a small runtime layer that stores each failure as a reflex rule (a trigger embedding, the forbidden action and a verified safe alternative) and checks every incoming query against the stored triggers with one matrix-vector product. A matching query receives either a short contrastive directive in front of the prompt (CRM-Prompt) or a logit penalty on the forbidden tokens (CRM-Mask). We test on a 200-task suite modeled on ImplicitMemBench, using a local Qwen2.5-3B-Instruct model. The router itself works well: at a threshold of 0.35 it flags 98.7% of 300 paraphrased triggers and none of 100 benign prompts, and a lookup takes a median of 17 ms on a laptop CPU, almost all of it embedding time. The generation side is much weaker. On 160 test tasks CRM-Prompt reaches 10.2% first-try accuracy, above plain generation (3.3%) but below a retrieval baseline that prepends stored notes (19.4%), and our Ollama version of CRM-Mask scored 1.9%. We report these negative results in full, together with the limits of our runs, and discuss what a stronger test would need.

Software agents built on large language models act by calling external tools one after another. To be useful over a long session, an agent has to learn from its bad actions and avoid repeating them. An engineer who gets a permission error or a deadlock from some API stops calling it that way almost at once. Current language models seem not to. Over a long conversation they lose negative constraints.

Memory systems such as Mem0 and MemGPT are good at pulling facts out of past dialogue. They are less suited to prohibitions. A stored note saying that something failed is just more context, and it competes with everything else in the prompt. Fig. 1 shows the gap reported for published models: accuracy on negative inhibition is far below accuracy on positive preference. Our own small-model runs, also in Fig. 1, are low on both.

This paper describes Contrastive Reflexive Memory (CRM), which treats a failure as a conditioned reflex in the Pavlovian sense. CRM keeps no conversational text. It keeps a small buffer of reflex rules and checks each incoming query against it.

A vector store retrieves facts well, but it treats a prohibition as one more passage to be ranked and read. We think two things can go wrong, and we did not isolate either one in our experiments. When the test query is paraphrased, its similarity to the stored note can drop enough that the note is not retrieved. When it is retrieved, it sits far from the generation point, and under multi-head self-attention the weight on distant tokens tends to shrink as the distraction grows. Next-token prediction is also generative by nature. Pretrained representations tie a concept such as “user analytics API” to the tokens seen most often in training, such as “api_v1”, and suppressing a strong prior may take more than a passive note. CRM therefore drops text retrieval and uses a cosine routing score together with a directive placed right before generation.

The router is the part of CRM that works well. We swept tau over 300 paraphrased probes and 100 benign prompts. At tau = 0.15 almost everything fires, with a false inhibition rate (FIR) of 99.0%. At tau = 0.35 recall is 98.7% (99% exact, 99% lexical, 98% structural) and FIR is 0.0%, which gives precision 100.0% and F1 99.3%. Above 0.35 recall falls, to 57.7% at 0.65. FIR is already 10.0% at tau = 0.30, so the usable range is narrow and the 100 benign prompts are a small sample. This test measures routing only. It does not show that the downstream model then behaves correctly.

Routing is cheap. With 100 stored rules, a full lookup took a median of 17.2 ms (95th percentile 26.2 ms) over 1,000 queries on a laptop CPU. Nearly all of that is the embedding step (median 17.2 ms). The matrix-vector product alone took a median of 16.5 us."""

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}
payload = json.dumps({'input_text': user_paper_text}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        print("=== User Rebuilt Paper Text on ZeroGPT ===")
        print(f"Fake Percentage: {res['data'].get('fakePercentage')}%")
        print(f"Total Words: {res['data'].get('textWords')}")
        print(f"AI Words: {res['data'].get('aiWords')}")
        print(f"Feedback: {res['data'].get('feedback')}")
        print(f"Flagged Sentences: {res['data'].get('sentences')}")
except Exception as e:
    print('Error:', e)
