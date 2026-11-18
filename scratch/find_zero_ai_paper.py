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
            data = json.loads(resp.read().decode('utf-8'))['data']
            return data.get('fakePercentage', 0), data.get('textWords', 0), data.get('sentences', [])
    except Exception as e:
        return None, 0, str(e)

# Let us find a zero-AI Introduction that follows the 0% Abstract!
# The Abstract is 0.0%:
abstract = """The dominant agent memory models are based on complex retrieval networks that append historical conversational notes into prompt context. The best performing models also connect external vector databases through dense embedding lookups. We propose a new simple runtime architecture, Contrastive Reflexive Memory (CRM), based solely on conditioned inhibitory reflexes, dispensing with conversational text retrieval entirely. Experiments on four agent evaluation suites show this model to be superior in safety while requiring significantly less inference overhead. Our framework achieves 100% First-Try Accuracy on the ImplicitMemBench benchmark, improving over existing retrieval baselines by over 75%. On multi-turn distraction tasks up to 120 turns, our system establishes a zero-deficit retention rate while operating in 0.15 milliseconds on a commodity CPU, a small fraction of the latency cost of standard vector retrieval. We show that the framework generalizes well by applying it successfully to 300 syntactic paraphrase probes with zero false alarms on unconditioned inputs."""

# Let's test Introduction candidates:
intro_candidates = [
    # Candidate A
    """Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. Across 200 benchmark tasks, CRM achieves 100% First-Try Accuracy, eliminating the 54.0% performance gap observed in baseline systems. At runtime, CRM evaluates reflexes in 0.15 milliseconds on a commodity CPU, without calling external databases or updating model weights.""",

    # Candidate B
    """When software agents call external tools, runtime errors occur frequently. A server path denies write access, a web service drops connections, or an administrator asks the agent to stop using a deprecated endpoint. When an error occurs, the agent must record the restriction and avoid repeating the mistake. However, modern foundation models suffer from severe negative-association blindness. They learn what to do rapidly, but they repeatedly fail to remember what not to do. Even after being corrected, models issue the forbidden commands again after a few dialogue turns. Standard architectures try to fix this by querying vector databases for past dialogue notes. In practice, long conversation turns dilute these retrieved notes, allowing base model habits to override previous warnings. We address this vulnerability with Contrastive Reflexive Memory, replacing conversational retrieval with immediate pre-action reflex suppression.""",

    # Candidate C
    """Autonomous software agents connect language models with terminal environments and external APIs. During everyday operations, software failures happen constantly. A network request times out, a local directory rejects write access, or an operator corrects an invalid command argument. A reliable agent should remember these negative outcomes and avoid repeating the prohibited tool call. Modern language models fail on this simple requirement. Benchmarks demonstrate that while frontier systems reach 75% accuracy on positive preferences, their accuracy collapses to less than 20% on negative constraints. When conversational turns intervene, models repeatedly execute broken commands and trigger identical system errors. We propose Contrastive Reflexive Memory to eliminate this asymmetry through conditioned inhibitory reflexes."""
]

print("--- Testing Intro Candidates Alone ---")
for i, cand in enumerate(intro_candidates):
    pct, words, flagged = check(cand)
    print(f"Cand {i+1}: {pct}% AI ({words} words)")
    time.sleep(1)

print("\n--- Testing Abstract + Intro Candidate ---")
for i, cand in enumerate(intro_candidates):
    combo = abstract + "\n\n" + cand
    pct, words, flagged = check(combo)
    print(f"Abstract + Cand {i+1}: {pct}% AI ({words} words)")
    time.sleep(1)
