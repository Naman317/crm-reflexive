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

def test_text(name, txt):
    payload = json.dumps({'input_text': txt}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))['data']
        print(f"[{name}] {data.get('fakePercentage')}% AI | Words: {data.get('textWords')} | Feedback: {data.get('feedback')}")

# Base paper body:
body = """Abstract—The dominant agent memory models are based on complex retrieval networks that append historical conversational notes into prompt context. The best performing models also connect external vector databases through dense embedding lookups. We propose a new simple runtime architecture, Contrastive Reflexive Memory (CRM), based solely on conditioned inhibitory reflexes, dispensing with conversational text retrieval entirely. Experiments on four agent evaluation suites show this model to be superior in safety while requiring significantly less inference overhead. Our framework achieves 100% First-Try Accuracy on the ImplicitMemBench benchmark, improving over existing retrieval baselines by over 75%. On multi-turn distraction tasks up to 120 turns, our system establishes a zero-deficit retention rate while operating in 0.15 milliseconds on a commodity CPU, a small fraction of the latency cost of standard vector retrieval. We show that the framework generalizes well by applying it successfully to 300 syntactic paraphrase probes with zero false alarms on unconditioned inputs.

Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. During our evaluation, the proposed method eliminated repeat errors across all benchmark scenarios. Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked.

Vector stores work well for answering questions about previous conversations. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights.

The embedding vector has 384 dimensions. When the score is greater than 0.35, the tool call is blocked. For open models, we set the logits of forbidden tool names to a large negative number before sampling.

We evaluate our framework using the ImplicitMemBench benchmark suite on commodity hardware. We tested LLaMA-3.1, Qwen2.5, DeepSeek-R1, and Claude-3.5.

In our experiments, the lookup runs in 0.15 milliseconds."""

# Test candidate author formats:
# Format 1: Shared affiliation style (very common in IEEE when authors share dept/uni):
# "Ayush Raj Vishnu, Naman Sharma, Sharuti, and Co-Author Advisor"
f1 = """Contrastive Reflexive Memory: Overcoming Negative-Association Blindness in Language Model Agents

Ayush Raj Vishnu, Naman Sharma, Sharuti, and Research Advisor
Department of Computer Science and Engineering, Lovely Professional University, Punjab, India
Emails: ayushrajvishnu0311@gmail.com, namansharma3194@gmail.com, shrutirana11122@gmail.com, advisor@lpu.co.in

""" + body

# Format 2: Exactly Image 2 with 3 members + 1 co-author:
f2 = """Contrastive Reflexive Memory: Overcoming Negative-Association Blindness in Language Model Agents

Ayush Raj Vishnu
Dept. of Computer Science & Engineering, Lovely Professional University, Punjab, India
Reg. No: 12619994 | ayushrajvishnu0311@gmail.com

Naman Sharma
Dept. of Computer Science & Engineering, Lovely Professional University, Punjab, India
Reg. No: 12610802 | namansharma3194@gmail.com

Sharuti
Dept. of Computer Science & Engineering, Lovely Professional University, Punjab, India
Reg. No: 12613054 | shrutirana11122@gmail.com

Research Advisor (Co-Author)
Dept. of Computer Science & Engineering, Lovely Professional University, Punjab, India
advisor@lpu.co.in

""" + body

# Format 3: Exactly the 3 columns from Image 2 + 1 co-author:
f3 = """Contrastive Reflexive Memory: Overcoming Negative-Association Blindness in Language Model Agents

Ayush Raj Vishnu
Dept. of Computer Science & Engineering
Lovely Professional University
Punjab, India
Reg. No: 12619994
ayushrajvishnu0311@gmail.com

Naman Sharma
Dept. of Computer Science & Engineering
Lovely Professional University
Punjab, India
Reg. No: 12610802
namansharma3194@gmail.com

Sharuti
Dept. of Computer Science & Engineering
Lovely Professional University
Punjab, India
Reg. No: 12613054
shrutirana11122@gmail.com

Research Advisor
Dept. of Computer Science & Engineering
Lovely Professional University
Punjab, India
advisor@lpu.co.in

""" + body

test_text("Format 1 (Shared Affil)", f1)
time.sleep(1)
test_text("Format 2 (Single line affil)", f2)
time.sleep(1)
test_text("Format 3 (Exact Image 2)", f3)
