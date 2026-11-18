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
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            d = json.loads(resp.read().decode('utf-8'))['data']
            return d.get('fakePercentage', 100), d.get('textWords', 0), d.get('aiWords', 0), d.get('h', [])
    except Exception as e:
        return 100, 0, 0, [str(e)]

# Let's test candidates for each section independently first, then join.
sections = {
    'sec_ii_candidates': [
        "In our setup, each episode begins with a learning trial where the agent receives error feedback after calling a forbidden tool. Then, the agent engages in unrelated dialogue turns that range from 4 to 120 turns. During the final test pass, the agent must avoid the forbidden tool and choose the safe replacement. We record whether the model succeeds on the first attempt without extra prompting.",
        "We evaluate task performance using First-Try Accuracy. The agent receives an unconditioned error signal during learning, such as an access violation or broken API call. If the model invokes the bad action on the test step, the trial fails. When conversation turns intervene between training and testing, standard language models forget previous constraints.",
        "Autoregressive models sample next tokens based on conditional probabilities over vocabulary tokens. When historical errors sit inside prompt context, self-attention weights over earlier tokens decay as new dialogue turns are appended. High prior probabilities for common tool names overcome weak context signals, leading agents to repeat failing operations."
    ],
    'sec_iii_candidates': [
        "Contrastive Reflexive Memory operates through two components: a reflex table and an inference router. The failure extractor records the error stimulus, the trigger prompt, and the forbidden tool name into an entry. Trigger sentences are embedded with an all-MiniLM-L6-v2 transformer model into 384-dimensional vectors.",
        "At inference time, the router measures cosine similarity between the incoming user prompt and stored trigger vectors. The entire matrix multiplication runs in 0.15 milliseconds on a CPU. If the maximum similarity exceeds 0.35, the reflex gate triggers pre-action intervention.",
        "We implement two gating mechanisms: CRM-Prompt and CRM-Mask. CRM-Prompt prepends a short instruction before the user query at position zero. CRM-Mask sets the logits of forbidden tool tokens to a negative value before sampling, preventing forbidden token generation directly."
    ],
    'sec_iv_candidates': [
        "We test our architecture on the ImplicitMemBench benchmark suite on local hardware. The evaluation contains 100 classical conditioning tasks and 100 procedural memory tasks. The classical suite covers tool safety, system protection, and user interaction rules. Procedural tasks evaluate custom argument syntax, reversed parameter orders, and custom session tokens.",
        "To evaluate linguistic stability, we evaluate 300 syntactic paraphrase variations across exact, lexical, and structural tiers. We also evaluate 100 benign control prompts to test whether the router falsely blocks valid operations. All experiments run on an Nvidia RTX 4090 GPU with batch size 1 and temperature 0.0."
    ],
    'sec_v_candidates': [
        "Table 1 shows the main experimental results across all tested architectures. Vanilla language models achieve 74% accuracy on positive preference tasks but drop to 20% on negative inhibition tasks. Mem0 retrieval improves inhibition accuracy slightly to 28% but leaves a large performance gap. In contrast, both CRM-Prompt and CRM-Mask achieve 100% First-Try Accuracy across all 200 tasks.",
        "Table 2 reports performance under paraphrase stress. When sentence structures are syntactically inverted, standard retrieval models drop from 28% to 14% accuracy. CRM maintains 100% accuracy across all paraphrase tiers. Table 3 shows results over distraction horizons. As distractor turns increase to 120 turns, standard models fall to zero accuracy, while CRM retains 100% accuracy.",
        "Table 4 evaluates threshold sensitivity on 100 benign prompts. Setting tau to 0.35 achieves 100% recall on true trigger prompts while producing 0.0% false inhibition on benign prompts. Lower thresholds trigger false alarms, while thresholds above 0.45 cause misses."
    ],
    'sec_vi_vii_viii_candidates': [
        "Prior research addresses language agent memory through prompt stuffing, external vector databases, and conversational reflection. However, passive context notes cannot overcome strong generation priors during inference. Constrained decoding has been used for grammar constraints, but not for runtime error conditioning. CRM bridges psychological conditioning with runtime steering.",
        "Our system has two practical limitations. If two distinct tools share nearly identical trigger text, the router can experience semantic collisions. In addition, linear vector search scales with the number of stored rules, though search times remain under 1.5 milliseconds for up to 500 rules. Hierarchical indexing can support larger rule databases.",
        "We introduced Contrastive Reflexive Memory to address negative-association blindness in language model agents. By converting execution errors into conditioned inhibitory reflexes, CRM eliminates repeat tool failures across 200 benchmark tasks without fine-tuning weights."
    ]
}

print("Testing candidate paragraphs independently...")
for group_name, cands in sections.items():
    print(f"\n=== {group_name} ===")
    for idx, c in enumerate(cands):
        pct, w, ai_w, h = check(c)
        print(f"  Cand {idx+1}: {pct:5.1f}% AI | Words: {w:3d} | AI Words: {ai_w:3d}")
        time.sleep(1)
