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

# Section III Lead-in:
sec3_lead = """To eliminate negative-association failures without retraining model weights, we propose Contrastive Reflexive Memory. The architecture operates through three modular components: failure extraction, semantic routing, and inhibitory actuation. This section details the mathematical mechanics of each stage."""

# Subsection III-A: Failure Extractor and Reflex Tuple Construction
sub_a_p1 = """Drawing inspiration from conditioned inhibition in classical psychology [9], [10], our architecture extracts execution failures from system feedback and error logs. Instead of saving long text notes into a database, we save a short embedding of the user prompt. When a tool fails, the system records the failure stimuli shown in (6). We map the prompt text into an embedding vector defined in (7)."""

# Subsection III-A variable descriptions:
sub_a_vars = """where P_CS denotes the prompt context, a_inhibit represents the forbidden tool call, and V_inhibit captures the recorded execution error from the environment."""
sub_a_vars7 = """where f_emb maps raw prompt tokens into a normalized dense representation with unit Euclidean length."""

# Subsection III-B: Semantic Router and Threshold Calibration
sub_b_p1 = """The router embeds incoming queries using transformer representations [8]. The system evaluates all stored rules through the matrix product formulated in (8). The winning rule corresponds to the maximum similarity index calculated in (9). When the score is greater than 0.35, the tool call is blocked. We calibrate this threshold across different values in Figure 3."""

sub_b_vars8 = """where E represents the stacked trigger embedding matrix and s denotes the resulting cosine similarity scores."""
sub_b_vars9 = """where k represents the index of the highest scoring reflex memory in the current library."""
sub_b_vars10 = """where tau is set to 0.35 to ensure reliable triggering across semantic paraphrases."""

# Subsection III-C: Inhibitory Actuation Mechanisms
sub_c_p1 = """We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models like LLaMA [12]. CRM-Prompt places an inhibitory instruction before the user prompt as shown in (11), leveraging in-context conditioning [11] without model retraining [7]. For open models, we set the logits of forbidden tool names to a large negative number before sampling using (12)."""

sub_c_vars11 = """where x_t denotes the raw user request and x_hat_t represents the modified prompt context delivered to the inference engine."""
sub_c_vars12 = """where z_k represents the unnormalized logit score for token k and M_k is the negative infinity mask applied to forbidden tool tokens."""

# Subsection III-D: Multi-Turn Execution Dynamics
sub_d_p1 = """To examine real-world agent behavior, we track multi-turn conversational sessions under extreme distraction. As illustrated in Figure 4, baseline models repeat forbidden actions after 10 distraction turns, whereas CRM reliably enforces safe tool execution across all 120 turns."""

tests = [
    ("Sec3 Lead", sec3_lead),
    ("Sub A P1", sub_a_p1),
    ("Sub A Vars 6", sub_a_vars),
    ("Sub A Vars 7", sub_a_vars7),
    ("Sub B P1", sub_b_p1),
    ("Sub B Vars 8", sub_b_vars8),
    ("Sub B Vars 9", sub_b_vars9),
    ("Sub B Vars 10", sub_b_vars10),
    ("Sub C P1", sub_c_p1),
    ("Sub C Vars 11", sub_c_vars11),
    ("Sub C Vars 12", sub_c_vars12),
    ("Sub D P1", sub_d_p1),
]

for name, text in tests:
    check(name, text)
    time.sleep(0.5)
