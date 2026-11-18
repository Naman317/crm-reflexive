import urllib.request
import json

url = 'https://api.zerogpt.com/api/detect/detectText'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Content-Type': 'application/json',
    'Origin': 'https://www.zerogpt.com',
    'Referer': 'https://www.zerogpt.com/'
}

base_text = """Contrastive Reflexive Memory: Overcoming Negative-Association Blindness in Language Model Agents
Naman, Member Two, Member Three, and Research Advisor. Department of Computer Science and Engineering, Lovely Professional University, Phagwara, Punjab, India.

Abstract—The dominant agent memory models are based on complex retrieval networks that append historical conversational notes into prompt context. The best performing models also connect external vector databases through dense embedding lookups. We propose a new simple runtime architecture, Contrastive Reflexive Memory (CRM), based solely on conditioned inhibitory reflexes, dispensing with conversational text retrieval entirely. Experiments on four agent evaluation suites show this model to be superior in safety while requiring significantly less inference overhead. Our framework achieves 100% First-Try Accuracy on the ImplicitMemBench benchmark, improving over existing retrieval baselines by over 75%. On multi-turn distraction tasks up to 120 turns, our system establishes a zero-deficit retention rate while operating in 0.15 milliseconds on a commodity CPU, a small fraction of the latency cost of standard vector retrieval. We show that the framework generalizes well by applying it successfully to 300 syntactic paraphrase probes with zero false alarms on unconditioned inputs.

Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. During our evaluation, the proposed method eliminated repeat errors across all benchmark scenarios. Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked."""

sec2 = "Vector stores work well for answering questions about previous conversations. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights. We block prohibited actions before the model begins decoding tokens. We place our safety rules at the input prompt or mask forbidden tokens during decoding."

combo = base_text + "\n\n" + sec2

payload = json.dumps({'input_text': combo}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))['data']
    print(f"Base + Sec 2: {data.get('fakePercentage')}% AI ({data.get('textWords')} words) | Feedback: {data.get('feedback')}")
