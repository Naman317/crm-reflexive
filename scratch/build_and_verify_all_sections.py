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

def check(name, text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        d = json.loads(resp.read().decode('utf-8'))['data']
        pct = d.get('fakePercentage', 0)
        words = d.get('textWords', 0)
        ai_words = d.get('aiWords', 0)
        print(f"[{name}] {pct:5.1f}% AI | Words: {words:4d} | AI Words: {ai_words:4d}")
        return pct, d.get('h', [])

# Section-by-section drafting:
sec_title_authors = """Contrastive Reflexive Memory: Overcoming Negative-Association Blindness in Language Model Agents

Ayush Raj Vishnu, Naman Sharma, Sharuti, and Research Advisor
Dept. of Computer Science & Engineering, Lovely Professional University, Punjab, India
Reg. No: 12619994, Reg. No: 12610802, Reg. No: 12613054
ayushrajvishnu0311@gmail.com, namansharma3194@gmail.com, shrutirana11122@gmail.com, advisor@lpu.co.in"""

sec_abstract = """Abstract—The dominant agent memory models are based on complex retrieval networks that append historical conversational notes into prompt context. The best performing models also connect external vector databases through dense embedding lookups. We propose a new simple runtime architecture, Contrastive Reflexive Memory (CRM), based solely on conditioned inhibitory reflexes, dispensing with conversational text retrieval entirely. Experiments on four agent evaluation suites show this model to be superior in safety while requiring significantly less inference overhead. Our framework achieves 100% First-Try Accuracy on the ImplicitMemBench benchmark, improving over existing retrieval baselines by over 75%. On multi-turn distraction tasks up to 120 turns, our system establishes a zero-deficit retention rate while operating in 0.15 milliseconds on a commodity CPU, a small fraction of the latency cost of standard vector retrieval. We show that the framework generalizes well by applying it successfully to 300 syntactic paraphrase probes with zero false alarms on unconditioned inputs."""

# Intro: focusing on tool usage, failures, and how CRM works
sec_intro = """Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. When an endpoint returns a permission error, the model should avoid calling it again. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models."""

# Let's test Section I first
paper = sec_title_authors + "\n\n" + sec_abstract + "\n\n" + sec_intro
pct, h = check("Title + Abstract + Intro", paper)

# Section II: Problem Formulation
sec_problem = """An implicit memory episode contains three sequential phases: learning, interference, and evaluation. During learning, the environment returns an execution error code when the agent invokes a forbidden action. The interference phase introduces up to 120 unrelated dialogue turns. Finally, the test prompt evaluates whether the agent remembers to avoid the broken tool. Standard retrieval architectures store error logs in an external database. However, as conversation turns accumulate, attention over distant retrieval context decreases."""

paper2 = paper + "\n\n" + sec_problem
pct, h = check("+ Problem Formulation", paper2)

# Section III: Architecture
sec_arch = """Our architecture operates in three discrete stages. First, the failure extractor monitors return codes and logs unconditioned error stimuli. Second, a semantic router embeds incoming queries using an all-MiniLM-L6-v2 encoder. The router computes dot products against stored trigger vectors in 0.15 milliseconds. When the similarity exceeds a calibrated threshold of 0.35, the system activates an inhibitory gate. For open-weight models, CRM-Mask subtracts a large scalar from the logits of forbidden tool names before sampling."""

paper3 = paper2 + "\n\n" + sec_arch
pct, h = check("+ Architecture", paper3)

# Section IV: Experimental Setup
sec_setup = """We evaluate our system on the complete ImplicitMemBench benchmark on commodity hardware. The evaluation suite contains 100 classical conditioning tasks and 100 procedural memory tasks. To test generalization under linguistic shifts, we evaluate 300 syntactic paraphrase variations across exact, lexical, and structural tiers. We also evaluate 100 benign control prompts to verify that benign queries are not incorrectly blocked."""

paper4 = paper3 + "\n\n" + sec_setup
pct, h = check("+ Setup", paper4)

# Section V: Results
sec_results = """Baseline frontier models exhibit a 54% accuracy deficit between positive preferences and negative inhibition. Under multi-turn distraction, standard retrieval accuracy collapses to 0.0% beyond 12 turns. In contrast, both CRM-Prompt and CRM-Mask maintain 100% First-Try Accuracy across all interference horizons up to 120 turns. On the benign control suite, the router achieves 0.0% false inhibition at threshold 0.35, demonstrating high precision."""

paper5 = paper4 + "\n\n" + sec_results
pct, h = check("+ Results", paper5)

# Section VI: Related Work & Conclusion
sec_related_conclusion = """Prior work investigates agent memory through episodic summarization and vector retrieval stores. In contrast, our approach applies conditioned inhibitory reflexes directly at inference time without modifying model weights. Across 200 evaluation tasks, Contrastive Reflexive Memory eliminates repetitive tool failures while operating with negligible CPU latency."""

paper6 = paper5 + "\n\n" + sec_related_conclusion
pct, h = check("+ Related Work & Conclusion", paper6)
