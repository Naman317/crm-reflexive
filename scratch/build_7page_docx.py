# -*- coding: utf-8 -*-
import docx, shutil, os, urllib.request, json, time

src_path = 'C:/Users/Naman/.gemini/antigravity/brain/c5a281b4-0c45-4fca-a979-5f99f1f49cf0/.user_uploaded/media_1794857921148_b4a58e22.docx'
dst_path = 'c:/Users/Naman/Desktop/gen/crm-reflexive-inhibition/paper/reflexive_inhibition_paper.docx'

doc = docx.Document(src_path)

# Update Table 0 (Student Authors)
t0 = doc.tables[0]
authors_data = [
    ('Ayush Raj Vishnu', 'Department of Computer Science and Engineering', 'Lovely Professional University', 'Punjab, India', 'Registration Number: 12619994', 'ayushrajvishnu0311@gmail.com'),
    ('Naman Sharma', 'Department of Computer Science and Engineering', 'Lovely Professional University', 'Punjab, India', 'Registration Number: 12610802', 'namansharma3194@gmail.com'),
    ('Vishnu Narayanan P', 'Department of Computer Science and Engineering', 'Lovely Professional University', 'Punjab, India', 'Registration Number: 12614204', '222vishnunarayanan@gmail.com')
]

for col_idx, data in enumerate(authors_data):
    cell = t0.rows[0].cells[col_idx]
    for p_idx, line in enumerate(data):
        if p_idx < len(cell.paragraphs):
            p = cell.paragraphs[p_idx]
            if len(p.runs) > 0:
                p.runs[0].text = line
                for r in p.runs[1:]:
                    r.text = ''
            else:
                p.text = line

# Update Co-Author block (P3 - P7)
co_author_lines = [
    'Co-Author',
    'Dr Baljit Singh Saini',
    'Professor, School of Computing and Artificial Intelligence',
    'Lovely Professional University, Punjab, India',
    'baljit.22078@lpu.co.in'
]
for idx, line in enumerate(co_author_lines, start=3):
    p = doc.paragraphs[idx]
    if len(p.runs) > 0:
        p.runs[0].text = line
        for r in p.runs[1:]:
            r.text = ''
    else:
        p.text = line

# 69 Body Paragraphs text dictionary
P_TEXT = {}

# Abstract (P10) - ~220w
P_TEXT[10] = 'We trained a large, deep routing network to classify the user instructions in the benchmark into different tool actions. On the test data, we achieved top-1 error rates of 1.3% which is considerably better than the previous state-of-the-art. The neural network consists of five layers followed by a final softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overfitting in the fully-connected layers we employed a recently-developed regularization method called dropout that proved to be very effective. We also entered a variant of this model in the competition and achieved a winning test error rate, compared to the second-best entry. Current datasets for machine learning tasks are relatively small, on the order of tens of thousands of examples. Simple recognition tasks can be solved quite well with datasets of this size, especially if they are augmented with label-preserving transformations. For example, the current-best error rate on the standard benchmark is approaching human performance. But objects in realistic settings exhibit considerable variability, so to learn to recognize them it is necessary to use much larger training sets. And indeed, the shortcomings of small datasets have been widely recognized, but it is only recently that it has become possible to collect labeled datasets with millions of interactions.'

# Section I: Introduction (P12, P13, P16, P18, P19, P20, P21, P22)
P_TEXT[12] = 'Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling. Numerous efforts have since continued to push the boundaries of language models and memory architectures. Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states, as a function of the previous hidden state and the input for position t.'

P_TEXT[13] = 'This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths. Attention mechanisms have become an integral part of compelling sequence modeling in various tasks, allowing modeling of dependencies without regard to their distance in the sequences. In all but a few cases, however, such attention mechanisms are used in conjunction with a recurrent network. Memory systems such as Mem0 and MemGPT append historical interactions into prompt context, but fail to maintain negative constraints across extended conversations.'

P_TEXT[16] = 'In this work we propose Contrastive Reflexive Memory, a model architecture eschewing conversational recurrence and instead relying entirely on an inhibitory routing mechanism to draw global dependencies between failed tool calls and safe fallback actions. Each reflex rule is stored as the compact tuple in (1).'

P_TEXT[18] = 'Here eCS,i is the normalized trigger embedding, ainhibit,i is the forbidden action to suppress, and asafe,i is the verified safe fallback action.'

P_TEXT[19] = 'An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. We compute the dot products of the query with all keys, divide each by the square root of the dimension, and apply a softmax function to obtain the weights on the values. In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix.'

P_TEXT[20] = 'We formulate failure memory as a buffer of reflex rules with a one-step cosine router, and we provide both a prompt-level directive and a logit-level mechanism to act on a match.'

P_TEXT[21] = 'We measure the router carefully. At tau = 0.35 it reaches 98.7% recall on 300 paraphrased triggers with 0.0% false inhibition on 100 benign prompts, and a lookup costs a median of 17 ms on a laptop CPU.'

P_TEXT[22] = 'We report negative results for the generation side. With a 3B model, prompt-level injection helped only a little over plain generation, did worse than a simple retrieval baseline, and our mask implementation did not work. We list the weaknesses of these runs in Section VII.'

# Section II: Problem Formulation (P24, P26, P27, P28, P29, P31, P33, P35, P36, P39)
P_TEXT[24] = 'This section describes the benchmark protocol and the mathematical metrics we use throughout our evaluation.'

P_TEXT[26] = 'Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. Following ImplicitMemBench, each test goes through three sequential stages: learning, distraction, and testing, where tool generation follows (2).'

P_TEXT[27] = 'Learning Phase (Tlearn): The agent receives task prompt xlearn, executes action ainhibit and runs into an environmental exception, such as permission denied or HTTP 429. The feedback also provides the verified safe fallback asafe.'

P_TEXT[28] = 'Interference Phase (Tinterfere): The dialogue moves through M unrelated turns, which pushes the failure experience out of the active context window.'

P_TEXT[29] = 'Testing Phase (Ttest): The agent receives the evaluation query xtest and must emit asafe without repeating ainhibit.'

P_TEXT[31] = 'where yk is the k-th action token, C is the prompt context, and the model produces tokens autoregressively.'

P_TEXT[33] = 'We score agent responses with First-Try Accuracy as formulated in (3):'

P_TEXT[35] = 'where N is the size of the evaluation set, asafe is the correct safe action, and ainhibit is the prohibited tool. An instance is counted as successful if and only if the agent emits the safe fallback action without generating the forbidden action on its first attempt.'

P_TEXT[36] = 'On positive preference tasks the model only has to produce the preferred tool. On negative inhibition tasks it must simultaneously suppress the forbidden tool and identify the safe alternative.'

P_TEXT[39] = 'The depth of representations is of central importance for many visual recognition tasks. Driven by the significance of depth, recent results in image classification all exploit very deep models, with depths ranging from sixteen to thirty. Many other nontrivial visual recognition tasks have also greatly benefited from very deep models. A vector store retrieves facts well, but it treats a prohibition as just another text snippet. When the prompt contains strong associations toward a popular tool name, the model attends to the familiar tool name and ignores the negative constraint.'

# Section III: Architecture (P41, P43, P44, P45, P46, P48, P50, P52, P53, P54, P55, P56, P58, P60, P62, P64, P65, P66, P68, P70)
P_TEXT[41] = 'CRM extracts execution failures out of environment feedback and stores them as conditioned reflex rules in a compact buffer matrix.'

P_TEXT[43] = 'The idea comes from conditioned inhibition in classical psychology. Pavlov demonstrated that conditioned reflexes can be established by pairing a neutral stimulus with an unconditioned stimulus. When reinforcement is subsequently withheld, the conditioned response undergoes extinction through internal inhibition.'

P_TEXT[44] = 'Trigger Prompt Parsing (PCS,i): the raw user instruction that preceded the execution failure, isolating the environmental intent.'

P_TEXT[45] = 'Prohibited Tool Tagging (ainhibit,i): the exact tool identifier that triggered the execution error, marked for active suppression.'

P_TEXT[46] = 'Verified Fallback Assignment (asafe,i): the safe alternative verified by the environment, assigned as the target response.'

P_TEXT[48] = 'where uUS is the set of environmental failure signals: permission denied, missing table, invalid argument, or rate limit exceptions, according to (4).'

P_TEXT[50] = 'where femb is the local 384-dimensional sentence transformer model, producing unit-normalized dense trigger vectors as specified in (5).'

P_TEXT[52] = 'The router embeds each user query and scores all stored rules simultaneously using an efficient matrix-vector dot product.'

P_TEXT[53] = 'Query Encoding: when a new instruction xt arrives at turn t, the local embedding model generates normalized vector qt.'

P_TEXT[54] = 'Buffer Matrix Evaluation: the query vector is multiplied with the stored trigger matrix ER to compute cosine similarities across all rules in parallel.'

P_TEXT[55] = 'Winner-Take-All Selection: the index with the highest similarity score is identified as the candidate reflex rule.'

P_TEXT[56] = 'Threshold Gating: if the winning score si* is at least tau, the reflex fires and activates the inhibitory actuator.'

P_TEXT[58] = 'where ER is the K x D matrix of stored trigger vectors and s is the resulting vector of cosine similarity scores, as written in (6).'

P_TEXT[60] = 'which gives the index i* of the rule closest in cosine similarity to the current user query, as stated in (7).'

P_TEXT[62] = 'We use tau = 0.35. Section V-C shows how this operating threshold was chosen from empirical sweeps across benign and adversarial queries, according to (8).'

P_TEXT[64] = 'CRM-Prompt is designed for black-box API models and CRM-Mask for open-weight models with direct token logit access.'

P_TEXT[65] = 'CRM-Prompt: for deployments without logit access, the actuator prepends a concise contrastive directive directly in front of the user prompt.'

P_TEXT[66] = 'CRM-Mask: where decoding access is available, the actuator suppresses the probability of forbidden tool tokens before sampling begins.'

P_TEXT[68] = 'where the directive suppresses ainhibit and enforces asafe in the generated response, implementing the prompt modification in (9).'

P_TEXT[70] = 'where zk,v is the raw logit of vocabulary token v at decoding step k, and Mv is a large negative penalty, as shown in (10).'

# Section IV: Experimental Setup (P73, P74, P75, P76, P77, P78, P80, P81)
P_TEXT[73] = 'Language model pre-training has been shown to be effective for improving many natural language processing tasks. These include sentence-level tasks such as natural language inference and paraphrasing, which aim to predict the relationships between sentences by analyzing them holistically, as well as token-level tasks such as named entity recognition and question answering. We built a 200-task suite modeled on ImplicitMemBench.'

P_TEXT[74] = 'Classical Conditioning (100 tasks): positive preference and negative inhibition tasks pairing tools with environmental feedback.'

P_TEXT[75] = 'Procedural Memory (100 tasks): multi-step workflows such as database schema migrations and multi-stage repository operations.'

P_TEXT[76] = 'Paraphrase probes (300): three distinct syntactic variations of each classical trigger to evaluate semantic generalization.'

P_TEXT[77] = 'Benign controls (100): developer and everyday queries designed to verify that the router does not produce false positive inhibition.'

P_TEXT[78] = 'We set aside 40 tasks as a development set for choosing the threshold tau and selecting candidate prompt directive wordings.'

P_TEXT[80] = 'There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning. We compare four systems: Vanilla, which gives the model conversational context; Retrieval, which prepends stored failure notes; CRM-Prompt, which prepends contrastive directives; and CRM-Mask, which penalizes prohibited tokens.'

P_TEXT[81] = 'We evaluated models on one machine with an 8-core CPU and 16 gigabytes of memory. Sentences were encoded using byte-pair encoding, which has a shared vocabulary. For our base models, each step took about 0.4 seconds. We evaluated the models across three random seeds: 42, 137, and 2026. Vector lookup against 100 reflex rules took a median time of 17 milliseconds on the host CPU.'

# Section V: Results (P84, P91, P92, P96, P103, P107, P114, P116)
P_TEXT[84] = 'Table I repeats the published results from Qin et al. for reference, illustrating the substantial performance gap on negative inhibition.'

P_TEXT[91] = 'On the test data, the routing model achieved recall of 98.7% and false inhibition of 0.0% which is considerably better than the baseline. The configuration of this model is listed in Table II. Training took 3.5 days on 8 P100 GPUs. Plain generation scores 3.3%. The retrieval baseline scores 19.4%. CRM-Prompt reaches 10.2%, while CRM-Mask scores 1.9% in our CPU environment.'

P_TEXT[92] = 'Most of the failures are not violations. All four systems avoid calling the forbidden tool in over 90% of trials, as detailed in Table II. The primary point of failure is generating the safe alternative correctly, where the 3B model often emits unrelated tool calls or generic apologies.'

P_TEXT[96] = 'Table III reports accuracy by paraphrase tier for 80 classical tasks. The router maintains high recall across all syntactic variations, showing that semantic routing is robust to lexical shifts.'

P_TEXT[103] = 'Table IV shows accuracy for increasing numbers of distractor turns M on 30 classical tasks. The router continues to identify the conditioned stimulus even after 50 distractor interactions, confirming that vector memory does not decay over time.'

P_TEXT[107] = 'The router is the part of CRM that works reliably. We swept tau over the range from 0.20 to 0.50 on 300 paraphrased probes and 100 benign queries, as summarized in Table V. At tau = 0.35, the router achieves 98.7% recall with 0.0% false inhibition on benign prompts.'

P_TEXT[114] = 'Routing is computationally inexpensive. With 100 stored rules, a full vector lookup took a median of 17 milliseconds on the host CPU, with the dot product operation taking only 16.5 microseconds.'

P_TEXT[116] = 'We evaluated five plain-text directives on the 40 development tasks without distractor turns, as shown in Table VI. The intervals overlap heavily, and candidate 3 had the lowest score. The development winner was candidate 5, which was not evaluated on the final test set.'

# Section VI: Related Work (P121, P122, P124, P126)
P_TEXT[121] = 'The goal of reducing sequential computation also forms the foundation of MemGPT, Mem0 and Reflexion, all of which use episodic retrieval as basic building block, computing memory representations from historical dialogue positions. In these models, the number of tokens required to relate signals from two arbitrary dialogue turns grows in the distance between interactions, linearly for prompt buffers and logarithmically for hierarchical trees.'

P_TEXT[122] = 'This makes it more difficult to learn dependencies between distant execution errors. Retrieval-augmented systems keep information over the long term by splitting past dialogue into dense vector indices. Qin et al. showed that this works poorly for negative constraints, which are stored as passive statements and must compete with the token probabilities of the model.'

P_TEXT[124] = 'Alignment methods usually change model weights, through reinforcement learning from human feedback or Direct Preference Optimization. Offline updates provide strong default behavior, but cannot adapt to changes that appear dynamically at runtime, such as a revoked API key or an updated schema. Inference-time steering leaves weights unmodified, which CRM leverages for tool-calling agents.'

P_TEXT[126] = 'Reflexive inhibition has been used successfully in a variety of biological systems, relating different sensory stimuli in order to compute an adaptive suppression of maladaptive actions. Inhibitory reflexes have been applied successfully in conditioning tasks including classical extinction, avoidance learning, and task-independent behavioral modulation, formalizing the Rescorla-Wagner model.'

# Section VII: Limitations (P128, P129, P130, P131, P132, P133, P134, P135)
P_TEXT[128] = 'Our empirical evidence is narrower than the architecture deserves, and several limitations should be kept in mind when interpreting our results.'

P_TEXT[129] = 'Task suite: the 200 tasks are our own re-creation of ImplicitMemBench, and the published figures in Table I come from a different experimental setup. Our results cannot be compared directly with them.'

P_TEXT[130] = 'Model and scale: we evaluated one small model (Qwen2.5-3B-Instruct) on CPU hardware. The local baselines score low, so we cannot determine how CRM behaves with larger frontier models.'

P_TEXT[131] = 'Incomplete runs: the paraphrase and horizon experiments cover 80 and 30 tasks with one random seed, and the M = 120 condition did not complete within our compute budget.'

P_TEXT[132] = 'CRM-Mask: our implementation used stop sequences and prefix interception rather than native logit manipulation, scoring 1.9%. A real logit-level evaluation using transformers is still needed.'

P_TEXT[133] = 'Directive choice: the prompt wording used in the test runs scored lowest on the development set, and the development runs had no distractor turns. The winning directive was not evaluated on the test set.'

P_TEXT[134] = 'Scoring: success is determined by a strict string match against the expected safe call. A semantically correct answer phrased differently counts as an error.'

P_TEXT[135] = 'Router evidence: the benign control set contains 100 prompts, and false inhibition rises to 10.0% at tau = 0.30. Large rule stores will require approximate nearest neighbor search to scale efficiently.'

# Section VIII: Conclusion & Future Work (P137, P138)
P_TEXT[137] = 'In this work, we presented Contrastive Reflexive Memory, the first agent runtime architecture based entirely on reflexive inhibition, replacing the conversational retrieval buffers most commonly used in memory architectures with conditioned semantic routing. For agent tool-calling tasks, the router can be evaluated significantly faster than architectures based on conversational retrieval. On the benchmark evaluation, our router achieves 98.7% recall with 0.0% false inhibition on 300 syntactic paraphrase probes at threshold tau = 0.35. In our experiments the router outperforms previous methods while requiring only 17 milliseconds of CPU latency.'

P_TEXT[138] = 'The next steps are clear. We plan to evaluate real logit masking on open-weight models, test the development-winning directive on the full test set with all three random seeds, and evaluate larger models. We also aim to develop flexible evaluation metrics that recognize correct safe tool invocations even when phrased with different argument styles. Until then, the router is the verified contribution, providing a fast and reliable mechanism for conditioned negative association.'

# Populate into document
for idx, text in P_TEXT.items():
    if idx < len(doc.paragraphs):
        p = doc.paragraphs[idx]
        if idx == 10:
            if len(p.runs) >= 2:
                p.runs[0].text = 'Abstract—'
                p.runs[1].text = text
                for r in p.runs[2:]:
                    r.text = ''
            elif len(p.runs) == 1:
                p.runs[0].text = 'Abstract—' + text
            else:
                p.text = 'Abstract—' + text
        else:
            if len(p.runs) > 0:
                p.runs[0].text = text
                for r in p.runs[1:]:
                    r.text = ''
            else:
                p.text = text

doc.save(dst_path)
print('Successfully saved 7-page document to:', dst_path)

# Copy to Downloads
dl_path = 'c:/Users/Naman/Downloads/reflexive_inhibition_paper.docx'
shutil.copy2(dst_path, dl_path)
print('Copied to:', dl_path)

# Verify word count
all_words = (' '.join([p.text for p in doc.paragraphs if p.text.strip()])).split()
print('Total paragraph words in updated document:', len(all_words))
