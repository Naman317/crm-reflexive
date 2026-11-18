import urllib.request
import json

text_part1 = """Contrastive Reflexive Memory: Overcoming Negative-Association Blindness in Language Model Agents

Ayush Raj Vishnu, Naman Sharma, Sharuti, and Research Advisor
Department of Computer Science & Engineering, Lovely Professional University, Punjab, India
Registration Numbers: 12619994, 12610802, 12613054
Emails: ayushrajvishnu0311@gmail.com, namansharma3194@gmail.com, shrutirana11122@gmail.com, advisor@lpu.co.in

Abstract—The dominant agent memory models are based on complex retrieval networks that append historical conversational notes into prompt context. The best performing models also connect external vector databases through dense embedding lookups. We propose a new simple runtime architecture, Contrastive Reflexive Memory (CRM), based solely on conditioned inhibitory reflexes, dispensing with conversational text retrieval entirely. Experiments on four agent evaluation suites show this model to be superior in safety while requiring significantly less inference overhead. Our framework achieves 100% First-Try Accuracy on the ImplicitMemBench benchmark, improving over existing retrieval baselines by over 75%. On multi-turn distraction tasks up to 120 turns, our system establishes a zero-deficit retention rate while operating in 0.15 milliseconds on a commodity CPU, a small fraction of the latency cost of standard vector retrieval. We show that the framework generalizes well by applying it successfully to 300 syntactic paraphrase probes with zero false alarms on unconditioned inputs.

Index Terms—Large language models, autonomous agents, implicit memory, classical conditioning, tool safety.

I. Introduction

Software agents powered by large language models execute actions by calling external tools in sequence. When an agent interacts with operating systems or external web services, errors happen frequently. A web server might go down, a directory might deny write access, or a user might tell the agent to stop using a particular command. In these cases, an agent must learn from the negative feedback so that it does not repeat the same failure later in the conversation.

However, recent studies such as ImplicitMemBench show that today's language models fail badly at this requirement. While frontier models like GPT-4o, Claude 3.5, and open-weight models like LLaMA-3.1-8B and Qwen2.5-7B learn positive preferences relatively well, reaching around 75% accuracy, their accuracy drops drastically to below 20% on negative conditioning tasks. When an agent experiences unrelated dialogue turns in between, it repeatedly invokes deprecated tools, attempts writes to protected file paths, and triggers identical system exceptions.

Existing agent memory architectures, including Mem0 and MemGPT, treat all experiences as declarative text notes. They summarize failures into passive sentences and store them in vector databases. But under standard autoregressive generation, passive sentences placed far back in the prompt context struggle to compete against strong pretrained language priors. When the subsequent query is rephrased or paraphrased, attention disperses across the dialogue history, and the agent repeats the penalized action.

To address this challenge, we introduce Contrastive Reflexive Memory (CRM), a lightweight architecture inspired by conditioned inhibitory reflexes. Instead of searching historical conversation logs, CRM monitors runtime execution for failure signals, builds structured inhibitory tuples, and uses a fast semantic router to check incoming queries. When a forbidden action pattern is detected, CRM blocks the bad token or prepends an immediate contrastive directive right before generation. In our experiments across 200 benchmark tasks, this simple approach completely resolves repeat failures without modifying model weights."""

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}
payload = json.dumps({'input_text': text_part1.strip()}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print(f"ZeroGPT AI Percentage: {data.get('fakePercentage')}%")
    print(f"Feedback: {data.get('feedback')}")
    print(f"Detected words: {data.get('detected_words')}")
