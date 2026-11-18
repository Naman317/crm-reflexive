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

def test_text(title, text):
    payload = json.dumps({'input_text': text.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))['data']
            fake = data.get('fakePercentage', 0.0)
            words = data.get('textWords', 0)
            fb = data.get('feedback', '')
            print(f"{title:<25} | Words: {words:>4} | Score: {fake:>5.1f}% | {fb}")
            return fake
    except Exception as e:
        print(f"{title:<25} | Error: {e}")
        return None

# Test sections from the user prompt:
sec1 = """Software agents built on large language models act by calling external tools one after another. To be useful over a long session, an agent has to learn from its bad actions and avoid repeating them. An engineer who gets a permission error or a deadlock from some API stops calling it that way almost at once. Current language models seem not to. Over a long conversation they lose negative constraints.

Memory systems such as Mem0 and MemGPT are good at pulling facts out of past dialogue. They are less suited to prohibitions. A stored note saying that something failed is just more context, and it competes with everything else in the prompt. Fig. 1 shows the gap reported for published models: accuracy on negative inhibition is far below accuracy on positive preference. Our own small-model runs, also in Fig. 1, are low on both.

This paper describes Contrastive Reflexive Memory (CRM), which treats a failure as a conditioned reflex in the Pavlovian sense. CRM keeps no conversational text. It keeps a small buffer of reflex rules and checks each incoming query against it.

We built two ways to act on a rule, CRM-Prompt for black-box models and CRM-Mask for open-weight models. Queries are embedded with a small sentence transformer (384 dimensions), and when the similarity score reaches 0.35 the reflex fires. CRM-Prompt then puts a short directive in front of the user query. CRM-Mask is meant to lower the logits of the forbidden tool names before sampling. We evaluate both on a 200-task suite modeled on ImplicitMemBench."""

sec2 = """Following ImplicitMemBench, each test goes through three stages: learning, distraction and testing. The agent makes an erroneous tool call and receives an error message. The user then asks a series of unrelated questions. At the end we check whether the agent avoids the broken action on its first attempt.

A vector store retrieves facts well, but it treats a prohibition as one more passage to be ranked and read. We think two things can go wrong, and we did not isolate either one in our experiments. When the test query is paraphrased, its similarity to the stored note can drop enough that the note is not retrieved. When it is retrieved, it sits far from the generation point, and under multi-head self-attention the weight on distant tokens tends to shrink as the distraction grows. Next-token prediction is also generative by nature. Pretrained representations tie a concept such as user analytics API to the tokens seen most often in training, such as api_v1, and suppressing a strong prior may take more than a passive note. CRM therefore drops text retrieval and uses a cosine routing score together with a directive placed right before generation."""

sec3 = """The router embeds each user query and scores all stored rules in one pass. It works in four steps:
1. Query Encoding: when a new instruction arrives, the router computes the normalized embedding.
2. Buffer Matrix Evaluation: the query vector is multiplied with the buffer matrix in a single matrix-vector product.
3. Winner-Take-All Selection: the index with the highest similarity score comes from the argmax.
4. Threshold Gating: if the winning score is at least tau, the gate activates the reflex. Below tau the gate stays at 0 and execution goes on unchanged.

We use tau = 0.35. Section V-C shows how this value was chosen and how narrow the margin is.

CRM-Prompt is meant for black-box models and CRM-Mask for open-weight models. CRM-Prompt puts an instruction in front of the user prompt. It relies on in-context conditioning and needs no retraining. CRM-Mask adds a large negative value to the logits of forbidden tool names before sampling."""

sec4 = """We built a 200-task suite modeled on ImplicitMemBench. It is our own re-creation and not the official release, so absolute numbers should not be compared with published ones. The suite has four parts:
1. Classical Conditioning (100 tasks): positive preference and negative inhibition tasks, covering API version restrictions, file and directory permissions, preferred communication tone and security boundaries.
2. Procedural Memory (100 tasks): multi-step habits such as reversed argument syntax, custom session tokens and non-standard output formats.
3. Paraphrase probes (300): three versions of each classical trigger, exact, lexical and structural.
4. Benign controls (100): developer and general queries that should not trigger any rule, used to measure false inhibition.

We set aside 40 tasks as a development set for choosing the directive wording and report results on the other 160. The two sets do not overlap.

Everything ran locally through Ollama on a laptop CPU with Qwen2.5-3B-Instruct, greedy decoding and a 128-token limit. Each test prompt follows 8 unrelated distractor turns, except in the horizon experiment."""

sec5 = """Plain generation scores 3.3%. The retrieval baseline scores 19.4%, the highest of the four, and CRM-Prompt reaches 10.2%. The paired comparison on identical task and seed pairs favors Mem0 over CRM-Prompt (67 tasks where only Mem0 succeeds, 23 where only CRM-Prompt does, McNemar p < 0.001) and CRM-Prompt over vanilla (40 against 7, p < 0.001). The gap comes mostly from procedural tasks, where Mem0 reaches 30.4% and CRM-Prompt 10.8%. On classical inhibition tasks the three intervals overlap, so we cannot tell the systems apart there.

Most of the failures are not violations. All four systems avoid the forbidden action in 76% to 92% of trials, and CRM-Mask avoids it most often (91.5%). What is rare is the safe action itself: CRM-Prompt writes it in 19.8% of trials and CRM-Mask in 1.9%. In practice the 3B model often explains, asks a question or refuses instead of making the call.

The router is the part of CRM that works well. We swept tau over 300 paraphrased probes and 100 benign prompts. At tau = 0.15 almost everything fires, with a false inhibition rate (FIR) of 99.0%. At tau = 0.35 recall is 98.7% (99% exact, 99% lexical, 98% structural) and FIR is 0.0%, which gives precision 100.0% and F1 99.3%. Above 0.35 recall falls, to 57.7% at 0.65. FIR is already 10.0% at tau = 0.30, so the usable range is narrow and the 100 benign prompts are a small sample."""

sec7 = """Our evidence is narrower than the architecture deserves, and the following points should be kept in mind when reading the results:
1. Task suite: the 200 tasks are our own re-creation of ImplicitMemBench, and the published numbers in Table I come from a different setup. Our results cannot be compared directly with them.
2. Model and scale: we used one small model (Qwen2.5-3B-Instruct) on a CPU. The local baselines score low, so we cannot say how CRM behaves with larger models.
3. Incomplete runs: the paraphrase and horizon experiments cover 80 and 30 tasks with one seed, and the M = 120 condition did not finish. We computed no retention rate. CRM-Prompt matched vanilla tier for tier in the paraphrase run, and we did not find out why.
4. CRM-Mask: our version used stop sequences and prefix interception, not logit editing, and it scored 1.9%. A real logit-level test, for example with Hugging Face transformers, is still missing.
5. Directive choice: the wording used in the test runs scored lowest on the development set, and the development runs had no distractor turns. The development winner was not run on the test set.
6. Scoring: success is a string match against the expected safe call. A correct answer in other words counts as a failure, and we did not audit a sample by hand.
7. Router evidence: the benign set has 100 prompts, FIR rises to 10.0% at tau = 0.30, and the probes were built from the same tasks as the stored triggers."""

sec8 = """We described Contrastive Reflexive Memory, a layer that turns tool failures into reflex rules and routes each new query against them with one matrix-vector product. The routing half held up well in our tests: 98.7% recall on paraphrased triggers with no false inhibition on benign prompts at tau = 0.35, at a cost of tens of microseconds for the product and about 17 ms with embedding. The generation half did not. On a 3B model, prompt-level injection improved on plain generation but not on a simple retrieval baseline, and our mask variant did not work in the form we could run.

The next steps are clear. We would run real logit masking, test the development-winning directive on the full test set with all three seeds and the paraphrase and horizon runs at full size, add a hand-audited sample of responses, and try larger models. We also want a scorer that recognizes the safe call when it is written in other words. Until then, the router is the result we trust, and the claim that CRM closes the negative-association gap remains unproven."""

print("Testing sections on ZeroGPT:")
test_text("Section 1 (Intro)", sec1)
time.sleep(1)
test_text("Section 2 (Problem)", sec2)
time.sleep(1)
test_text("Section 3 (Architecture)", sec3)
time.sleep(1)
test_text("Section 4 (Setup)", sec4)
time.sleep(1)
test_text("Section 5 (Results)", sec5)
time.sleep(1)
test_text("Section 7 (Limitations)", sec7)
time.sleep(1)
test_text("Section 8 (Conclusion)", sec8)
