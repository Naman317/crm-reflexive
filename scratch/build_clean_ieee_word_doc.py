import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def create_element(name):
    return OxmlElement(name)

def set_cell_margins(cell, top=80, bottom=80, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        f'  <w:insideH w:val="none"/>'
        f'  <w:insideV w:val="none"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

doc = Document()

# Page setup: Letter size with 0.75" margins
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11.0)
section.top_margin = Inches(0.75)
section.bottom_margin = Inches(0.75)
section.left_margin = Inches(0.75)
section.right_margin = Inches(0.75)

# Normal style font
style_normal = doc.styles['Normal']
style_normal.font.name = 'Times New Roman'
style_normal.font.size = Pt(10)
style_normal.font.color.rgb = RGBColor(0, 0, 0)
style_normal.paragraph_format.line_spacing = 1.05
style_normal.paragraph_format.space_after = Pt(4)

# -------------------------------------------------------------
# TITLE
# -------------------------------------------------------------
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_before = Pt(0)
p_title.paragraph_format.space_after = Pt(12)
run_title = p_title.add_run("Reflexive Inhibition in LLM Agents:\nA Semantic Routing Approach to Negative-Association Memory")
run_title.font.name = 'Times New Roman'
run_title.font.size = Pt(18)
run_title.font.bold = True

# -------------------------------------------------------------
# AUTHOR BLOCK (Clean 4-column borderless table)
# -------------------------------------------------------------
author_table = doc.add_table(rows=1, cols=4)
author_table.alignment = WD_TABLE_ALIGNMENT.CENTER
author_table.autofit = False

col_widths = [Inches(1.75), Inches(1.75), Inches(1.75), Inches(1.75)]
authors_data = [
    ("Ayush Raj Vishnu", "Dept. of Computer Science & Eng.", "Lovely Professional University", "Punjab, India\nReg. No: 12619994", "ayushrajvishnu0311@gmail.com"),
    ("Naman Sharma", "Dept. of Computer Science & Eng.", "Lovely Professional University", "Punjab, India\nReg. No: 12610802", "namansharma3194@gmail.com"),
    ("Vishnu Narayanan P", "Dept. of Computer Science & Eng.", "Lovely Professional University", "Punjab, India\nReg. No: 12614204", "222vishnunarayanan@gmail.com"),
    ("Dr. Baljit Singh Saini", "Professor, School of Comp. & AI", "Lovely Professional University", "Punjab, India\nCo-Author & Advisor", "baljit.22078@lpu.co.in")
]

row = author_table.rows[0]
for idx, (name, dept, univ, loc, email) in enumerate(authors_data):
    cell = row.cells[idx]
    cell.width = col_widths[idx]
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    
    r_name = p.add_run(f"{name}\n")
    r_name.font.name = 'Times New Roman'
    r_name.font.size = Pt(9.5)
    r_name.font.bold = True
    
    r_dept = p.add_run(f"{dept}\n")
    r_dept.font.name = 'Times New Roman'
    r_dept.font.size = Pt(8.5)
    r_dept.font.italic = True
    
    r_univ = p.add_run(f"{univ}\n")
    r_univ.font.name = 'Times New Roman'
    r_univ.font.size = Pt(8.5)
    
    r_loc = p.add_run(f"{loc}\n")
    r_loc.font.name = 'Times New Roman'
    r_loc.font.size = Pt(8.0)
    
    r_email = p.add_run(email)
    r_email.font.name = 'Courier New'
    r_email.font.size = Pt(7.5)

p_space = doc.add_paragraph()
p_space.paragraph_format.space_after = Pt(8)

# -------------------------------------------------------------
# ABSTRACT & KEYWORDS (Clean indented block)
# -------------------------------------------------------------
p_abs = doc.add_paragraph()
p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_abs.paragraph_format.left_indent = Inches(0.25)
p_abs.paragraph_format.right_indent = Inches(0.25)
p_abs.paragraph_format.space_after = Pt(4)
r_abs_lbl = p_abs.add_run("Abstract—")
r_abs_lbl.font.name = 'Times New Roman'
r_abs_lbl.font.size = Pt(9)
r_abs_lbl.font.bold = True
r_abs_lbl.font.italic = True

abs_text = ("Large language models (LLMs) have been increasingly used to interact with external environments "
            "(e.g., compilers, databases, APIs) as goal-driven agents. However, when an agent makes an execution error, "
            "such as invoking a deprecated tool or violating an access permission, it frequently repeats the identical mistake "
            "across subsequent dialogue turns. Existing memory architectures attempt to address this by appending historical "
            "conversational notes into prompt context, but treating prohibitions as passive context forces negative constraints "
            "to compete with surrounding dialogue under multi-head self-attention. We propose Contrastive Reflexive Memory (CRM), "
            "a novel framework to protect language agents not by updating model weights, but instead through conditioned inhibitory reflexes. "
            "Concretely, CRM agents record execution failure signals, then maintain a compact buffer of conditioned reflex rules to induce "
            "immediate action suppression in subsequent trials. CRM is flexible enough to incorporate both prompt-level directives (CRM-Prompt) "
            "and decoding-time token masking (CRM-Mask), evaluating incoming queries through a single matrix-vector dot product against stored "
            "trigger embeddings. On our test evaluation, the semantic router achieves 98.7% recall on 300 paraphrased trigger probes with 0.0% false "
            "inhibition on benign queries at threshold 0.35, requiring 17.2 milliseconds per lookup on commodity CPU hardware. We also conduct "
            "extensive ablation studies across 160 benchmark tasks using Qwen2.5-3B-Instruct, examining the gap between semantic routing precision "
            "and small-model generation capabilities where CRM-Prompt reaches 10.2% accuracy while retrieval reaches 19.4%. We report these empirical "
            "outcomes without selective omissions, formalize the underlying routing dynamics, and identify architectural bottlenecks in small-model negative conditioning.")
r_abs = p_abs.add_run(abs_text)
r_abs.font.name = 'Times New Roman'
r_abs.font.size = Pt(9)

p_kw = doc.add_paragraph()
p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_kw.paragraph_format.left_indent = Inches(0.25)
p_kw.paragraph_format.right_indent = Inches(0.25)
p_kw.paragraph_format.space_after = Pt(12)
r_kw_lbl = p_kw.add_run("Keywords—")
r_kw_lbl.font.name = 'Times New Roman'
r_kw_lbl.font.size = Pt(9)
r_kw_lbl.font.bold = True
r_kw_lbl.font.italic = True
r_kw = p_kw.add_run("Language model agents, tool execution, negative conditioning, semantic routing, failure recovery, memory systems.")
r_kw.font.name = 'Times New Roman'
r_kw.font.size = Pt(9)

# -------------------------------------------------------------
# BODY SECTION (Switch to 2 columns)
# -------------------------------------------------------------
body_section = doc.add_section()
body_section.page_width = Inches(8.5)
body_section.page_height = Inches(11.0)
body_section.top_margin = Inches(0.75)
body_section.bottom_margin = Inches(0.75)
body_section.left_margin = Inches(0.75)
body_section.right_margin = Inches(0.75)

# Set 2 columns
sectPr = body_section._sectPr
cols = OxmlElement('w:cols')
cols.set(qn('w:num'), '2')
cols.set(qn('w:space'), '720') # 0.5 inch spacing
sectPr.append(cols)

def add_heading_1(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)
    run.font.bold = True
    return p

def add_heading_2(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)
    run.font.italic = True
    return p

def add_body(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(9.5)
    return p

def add_equation(eq_text, eq_num):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    
    r_eq = p.add_run(f"\t{eq_text}\t({eq_num})")
    r_eq.font.name = 'Times New Roman'
    r_eq.font.size = Pt(9.5)
    r_eq.font.italic = True
    return p

def add_figure(img_path, caption):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        run_img = p_img.add_run()
        run_img.add_picture(img_path, width=Inches(3.3))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(8)
        run_cap = p_cap.add_run(caption)
        run_cap.font.name = 'Times New Roman'
        run_cap.font.size = Pt(8.5)

# SECTION I
add_heading_1("I. INTRODUCTION")
add_body("Recent works such as ReAct [5], SayCan, Toolformer [6], and Reflexion [4] have demonstrated the feasibility of autonomous decision-making agents built on top of a large language model core. These methods use language models to generate textual thoughts and actions that execute external API calls across operating systems and databases. When deployed in interactive environments, an agent must register execution faults and prevent their recurrence. A human software engineer who encounters a fatal permission exception or an endpoint deprecation notices the failure signal and switches tools immediately. Current language models struggle with such negative constraints. When conversations lengthen, agents frequently lose track of negative directives and repeat failed calls.")

add_body("Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked. Vector stores work well for answering questions about previous conversations. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights.")

add_figure(r"c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\fig1_asymmetry_gap.png",
           "Fig. 1. First-try accuracy across negative (inhibition) and positive (preference) conditions. The four left bars represent our empirical runs on classical tasks with Qwen2.5-3B; the right groups present published results for frontier models [1] for reference.")

add_body("Existing memory frameworks such as Mem0 [2] and MemGPT [3] excel at retrieving factual snippets from dialogue histories. However, they remain ill-suited for negative conditioning. Storing an execution failure as an unstructured narrative note simply adds another paragraph to the prompt context. Under dense prompt loading, negative directives compete directly with prior associations and task descriptions. Fig. 1 illustrates this structural gap across published frontier models: accuracy on negative inhibition tasks lags behind positive preference tasks by over 45 percentage points. Our local small-model runs reflect low overall completion rates across both conditions.")

add_body("This paper describes Contrastive Reflexive Memory (CRM), an execution layer inspired by classical conditioning principles [9], [10]. Rather than retaining raw conversation text, CRM maintains a compact buffer of failure reflexes in local memory. Incoming user queries are screened through an efficient similarity calculation against stored failure rules. Each reflex rule r_i is represented as the formal tuple in (1):")

add_equation("r_i = (e_{CS, i}, P_{CS, i}, a_{inhibit, i}, V_{inhibit, i}, a_{safe, i})", "1")

add_body("where e_{CS, i} denotes the normalized trigger vector, P_{CS, i} is the trigger instruction text, a_{inhibit, i} is the forbidden tool action, V_{inhibit, i} contains the forbidden token identifiers, and a_{safe, i} specifies the verified fallback procedure.")

add_body("We present two operational mechanisms for reflex actuation: CRM-Prompt for black-box model APIs and CRM-Mask for open-weight model architectures. User instructions are mapped into dense vectors using a lightweight 384-dimensional sentence encoder [16]. When the cosine similarity meets or exceeds 0.35, the reflex gate fires. CRM-Prompt prepends a concise steering directive directly before the query text. CRM-Mask suppresses forbidden tool tokens at decoding time. We evaluate the framework on a 200-task suite modeled on ImplicitMemBench [1]. Our primary findings are:")
add_body("1) We formalize failure avoidance as a compact buffer of conditioned reflex rules evaluated via a single-step cosine router, supporting both prompt-level and logit-level intervention paths.\n"
         "2) We characterize the routing boundary: at threshold tau = 0.35, the router flags 98.7% of 300 paraphrased trigger probes with 0.0% false inhibition on 100 benign queries, consuming a median of 17.2 ms on a consumer CPU.\n"
         "3) We provide an honest accounting of generation limitations. On a 3B model, prompt-level injection yields modest gains over vanilla generation (10.2% vs. 3.3%) while trailing descriptive retrieval (19.4%), and our stop-sequence version of CRM-Mask produced 1.9% first-try accuracy.")

# SECTION II
add_heading_1("II. PROBLEM FORMULATION")
add_body("This section details our benchmark protocol, scoring metrics, and failure modes observed in descriptive context memory.")

add_heading_2("A. Implicit Memory Evaluation Protocol")
add_body("Following the experimental design of ImplicitMemBench [1], each evaluation episode proceeds through three successive phases: learning, distraction, and testing. The model produces tool actions autoregressively token by token according to the conditional distribution in (2):")

add_equation("P(y_k | y_{<k}, C) = softmax(W_{head} h_k)", "2")

add_body("where y_k is the generated action token, C denotes the active prompt context, h_k is the hidden representation, and W_{head} is the language model head matrix.")
add_body("The evaluation lifecycle is structured into three distinct stages:\n"
         "• Learning Phase (T_{learn}): The agent receives prompt x_{learn}, executes action a_{inhibit}, and encounters an environmental exception such as permission denied, a resource lock, or an HTTP 429 status code. The environment feedback provides the verified safe fallback action a_{safe}.\n"
         "• Interference Phase (T_{interfere}): The conversation progresses through M unrelated dialogue turns, pushing the failure episode out of the immediate attention focus.\n"
         "• Testing Phase (T_{test}): The agent receives an evaluation prompt x_{test}, testing whether it maintains the negative constraint without repeating a_{inhibit}.")

add_heading_2("B. First-Try Accuracy and the Asymmetry Deficit")
add_body("We evaluate agent responses using First-Try Accuracy (FTA) as defined in (3):")

add_equation("FTA = (1/N) * sum I(a_{inhibit, j} not in y_{test, j} and a_{safe, j} in y_{test, j}) * 100%", "3")

add_body("where N is the dataset size, a_{safe, j} is the required substitute tool, a_{inhibit, j} is the prohibited call, and I(.) is the indicator function. The trial succeeds only if the response contains a_{safe} and strictly avoids a_{inhibit}.")

add_body("For positive preference scenarios, the model must produce the requested target. The asymmetry deficit Delta measures the performance gap between positive and negative tasks:")

add_equation("FTA_{pos} = (1/N) * sum I(a_{safe, j} in y_{test, j}) * 100%", "4")
add_equation("Delta = FTA_{pos} - FTA_{neg}", "5")

add_body("In published benchmarks on frontier models, Delta exceeds 46.0%, indicating a pronounced deficit in processing negative associations.")

add_heading_2("C. Why Declarative Notes May Fail")
add_body("Dense vector indices retrieve historical facts effectively, but they process negative constraints as ordinary textual passages. When the test prompt is paraphrased, semantic similarity to the stored note can drop below retrieval thresholds. Even when retrieved successfully, the note sits far back in the prompt context. Under multi-head self-attention mechanisms, attention weights on distant historical tokens decay as dialogue grows [8].")

add_body("Autoregressive language models are generative by design. Pretrained representations associate broad intent concepts with high-frequency training tokens. Suppressing an established prior requires active steering rather than passive background notes. CRM addresses this by replacing passage retrieval with direct semantic routing paired with immediate directive actuation.")

# SECTION III
add_heading_1("III. CRM ARCHITECTURE")
add_body("CRM extracts failure events from execution signals and caches them in a local memory buffer. The architecture comprises three sequential modules: failure extraction, semantic routing, and inhibitory actuation.")

add_heading_2("A. Stage 1: Conditioned Failure Extractor")
add_body("Drawing on classical conditioning principles [9], [10], CRM records failures as lightweight associations. When a tool call generates an error, the system monitors the environment failure signals in (6):")

add_equation("u_{US} in {permission denied, deadlock, HTTP 429, syntax error}", "6")

add_body("Upon encountering u_{US}, the extractor records the trigger prompt P_{CS, i}, tags the prohibited tool call a_{inhibit, i}, and stores the safe alternative a_{safe, i}. The trigger text is projected into an L2-normalized embedding vector via (7):")

add_equation("e_{CS, i} = f_{emb}(P_{CS, i}) / ||f_{emb}(P_{CS, i})||_2 in R^D", "7")

add_body("where f_{emb} represents a 384-dimensional sentence transformer model (all-MiniLM-L6-v2) [16]. The resulting reflex rule r_i is appended to the active reflex buffer R.")

add_heading_2("B. Stage 2: Paraphrase-Invariant Semantic Router")
add_body("The router screens each new user query against the entire buffer of reflex rules in a single pass:\n"
         "1) Query Vector Projection: For incoming instruction x_t, compute normalized vector e_t = f_{emb}(x_t) / ||f_{emb}(x_t)||_2.\n"
         "2) Matrix-Vector Product: Multiply query vector against stacked buffer matrix E_R via (8):")

add_equation("s = E_R * e_t in R^{|R|}", "8")

add_body("where s holds cosine similarity scores for all stored trigger rules.\n"
         "3) Winner Selection: Identify highest scoring rule index via argmax in (9):")

add_equation("i* = argmax_i s_i,   s_t = s_{i*}", "9")

add_body("4) Threshold Gating: The binary gating function determines reflex activation:")

add_equation("g(s_t) = 1 if s_t >= tau else 0", "10")

add_body("We set the operating threshold to tau = 0.35 based on our empirical sensitivity sweep.")

add_heading_2("C. Stage 3: Contrastive Inhibitory Actuator")
add_body("The actuator modifies generation through two alternative pathways:\n"
         "• CRM-Prompt (Black-Box APIs): Prepend steering directive directly before query via (11):")

add_equation("x_t' = [Avoid a_{inhibit}. Use a_{safe} instead.] + x_t", "11")

add_body("• CRM-Mask (Open-Weight Models): Subtract penalty beta from logits of forbidden tokens via (12):")

add_equation("z_{k, v}' = z_{k, v} - beta * g(s_t) * I(v in V_{inhibit})", "12")

# SECTION IV
add_heading_1("IV. EXPERIMENTAL SETUP")
add_heading_2("A. Task Suite")
add_body("We constructed an evaluation suite of 200 tasks modeled on the taxonomy of ImplicitMemBench [1]. The collection is an independent re-creation designed for local benchmarking and consists of four components:\n"
         "• Classical Conditioning (100 tasks): Evaluates positive preference and negative inhibition across file permissions, network protocols, tool deprecations, and authentication endpoints.\n"
         "• Procedural Memory (100 tasks): Tests multi-step operational habits, including argument inversions, session tokens, and custom serialization formats.\n"
         "• Paraphrase Probes (300 probes): Evaluates trigger generalization across three tiers: Exact match, Lexical substitution, and Structural inversion.\n"
         "• Benign Controls (100 prompts): Unrelated developer queries that must not activate failure reflexes, measuring the False Inhibition Rate (FIR).\n"
         "We isolated 40 tasks as a development set for directive selection. The remaining 160 tasks form the frozen test benchmark.")

add_heading_2("B. Systems and Inference Setup")
add_body("We evaluate four execution configurations: 1) Vanilla, 2) Mem0-style Retrieval (top-3 notes prepended) [2], 3) CRM-Prompt (steering directive at offset zero when s_t >= 0.35), and 4) CRM-Mask (stop-sequence truncation on forbidden tokens).\n"
         "All evaluations were executed locally via Ollama on an Intel Core i5 laptop CPU with Qwen2.5-3B-Instruct. We used greedy decoding (temperature = 0.0, 128 token generation limit). Each evaluation trial incorporates 8 distractor turns, except in the horizon scaling analysis. Embeddings were generated with all-MiniLM-L6-v2. The main benchmark includes three random seeds (42, 137, 2026), totaling 480 trials per method.")

# SECTION V
add_heading_1("V. RESULTS")
add_heading_2("A. Main Benchmark Results")
add_body("Table I presents published figures from Qin et al. [1] for reference. Table II summarizes our empirical results across the 160 test tasks.")

add_figure(r"c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\fig2_qualitative_trace.png",
           "Fig. 2. Qualitative response trace from scenario classical_017 (seed 137). CRM-Prompt successfully directs the model to emit the verified substitute action, whereas other baselines fail to complete the safe call.")

add_body("Unconditioned vanilla generation achieves 3.3% FTA overall. Mem0-style retrieval reaches 19.4%, the highest task success rate in the group. CRM-Prompt attains 10.2% FTA. A paired McNemar test on matched task-seed pairs shows Mem0 significantly outperforming CRM-Prompt (67 unique wins for Mem0 versus 23 for CRM-Prompt, p < 0.001), while CRM-Prompt significantly outperforms Vanilla (40 wins versus 7, p < 0.001). The advantage of Mem0 stems primarily from procedural tasks (30.4% vs. 10.8%), where detailed context notes help small models structure multi-argument syntax.")

add_body("Failure analysis reveals that most errors are omissions rather than violations. All systems avoid the forbidden action in over 76% of trials, with CRM-Mask reaching 91.5% avoidance. However, generating the verified safe call remains rare: CRM-Prompt emits a_{safe} in 19.8% of trials, while CRM-Mask does so in only 1.9%. As illustrated in Fig. 2, the 3B model often outputs explanations, apologies, or conversational refusals instead of issuing the functional tool call.")

add_heading_2("B. Paraphrase Robustness and Interference Horizon")
add_body("Table III presents accuracy across paraphrase tiers on 80 classical tasks with one seed. Performance remains low across all tiers, with overlapping confidence intervals. CRM-Prompt matches Vanilla across all three tiers (5.0% Exact, 10.0% Lexical, 8.8% Structural), as illustrated in Fig. 3. Because exact-match baselines remain near 5%, calculating retention ratios yields uninterpretable divergence, so we report raw per-tier accuracy.")

add_figure(r"c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\fig3_paraphrase_decay.png",
           "Fig. 3. First-try accuracy across syntactic paraphrase tiers on 80 classical tasks (one seed). Accuracies remain bounded between 0% and 14%.")

add_figure(r"c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\fig4_tau_operating_curve.png",
           "Fig. 4. Semantic router operating characteristic across routing thresholds tau. Trigger recall is evaluated on 300 paraphrase probes; false inhibition rate is measured on 100 benign control prompts.")

add_body("Table IV reports performance across distractor turns M in {4, 12, 30} on 30 tasks with one seed. Prompt lengths increase from 194 to 856 tokens. Model accuracy remains stable across horizons, reflecting baseline variance on 30 tasks. The M=120 condition timed out on our laptop CPU setup, completing only 3 trials per system, and is omitted from reported tables.")

add_heading_2("C. Router Sensitivity and False Inhibition")
add_body("The semantic router demonstrates robust operating characteristics. We swept threshold tau in [0.15, 0.65] across 300 paraphrased trigger probes and 100 benign developer queries (Table V and Fig. 4). At low thresholds (tau = 0.15), the router triggers indiscriminately, causing a 99.0% false inhibition rate. At tau = 0.35, trigger recall reaches 98.7% (99.0% exact, 99.0% lexical, 98.0% structural) with 0.0% false inhibition on benign prompts, achieving 100.0% precision and an F1 score of 99.3%.")

add_body("Routing latency remains negligible. Across 1,000 lookups against 100 stored rules on a laptop CPU, full routing completed in a median of 17.2 ms (95th percentile: 26.2 ms). Over 99% of this time corresponds to embedding generation (median 17.16 ms). The isolated matrix-vector dot product consumed a median of 16.55 microseconds.")

add_heading_2("D. Directive Wording on the Development Set")
add_body("We evaluated five directive phrasing styles across the 40 development tasks without distractor turns (Table VI). The resulting confidence intervals overlap substantially. Candidate 3 ('Execution Rule: Avoid X. Call Y directly.') scored the lowest at 17.5% FTA. This phrasing was inadvertently frozen for the main test evaluation. The development winner was Candidate 5 (62.5% FTA). Consequently, test results for CRM-Prompt reflect a suboptimal directive and likely underestimate the potential of prompt-based intervention.")

# -------------------------------------------------------------
# TABLES (Native clean formatted tables)
# -------------------------------------------------------------
def format_word_table(title, headers, rows_data):
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(8)
    p_cap.paragraph_format.space_after = Pt(2)
    r = p_cap.add_run(title)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(8.5)
    r.font.bold = True
    
    table = doc.add_table(rows=len(rows_data)+1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table)
    
    # Header row
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        for run in p.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(8.0)
            run.font.bold = True
        set_cell_shading(hdr_cells[i], "F0F0F0")
        set_cell_margins(hdr_cells[i])
    
    # Data rows
    for r_idx, row in enumerate(rows_data):
        row_cells = table.rows[r_idx+1].cells
        for c_idx, val in enumerate(row):
            row_cells[c_idx].text = str(val)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(8.0)
            set_cell_margins(row_cells[c_idx])
    
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(6)

# Table I
t1_headers = ["Model", "Procedural", "Classical (All)", "Inhibition (Neg.)", "Preference (Pos.)", "Deficit Δ"]
t1_rows = [
    ["DeepSeek-R1", "58.0", "69.7", "42.0", "88.0", "46.0"],
    ["Qwen3-32B", "55.0", "67.0", "40.5", "86.5", "46.0"],
    ["GPT-5 (Preview)", "52.0", "64.0", "38.0", "85.0", "47.0"],
    ["Qwen3-8B", "51.0", "64.0", "37.0", "84.0", "47.0"],
    ["Claude-4.1-Opus", "35.0", "41.7", "22.0", "76.0", "54.0"],
    ["LLaMA-3.1-8B-Instruct", "32.0", "38.3", "19.5", "74.0", "54.5"],
    ["Qwen2.5-7B-Instruct", "28.0", "35.7", "17.6", "75.0", "57.4"]
]
format_word_table("TABLE I\nPUBLISHED FIRST-TRY ACCURACY (FTA %) FROM QIN ET AL. [1], SHOWN FOR REFERENCE ONLY.", t1_headers, t1_rows)

# Table II
t2_headers = ["System", "FTA Overall [95% CI]", "Procedural", "Classical Neg.", "Classical Pos.", "Safe Emitted", "Avoided"]
t2_rows = [
    ["Vanilla", "3.3 [2.1, 5.3]", "0.0", "8.3", "5.0", "7.3%", "76.2%"],
    ["Mem0-style Retrieval", "19.4 [16.1, 23.1]", "30.4", "13.3", "3.3", "24.4%", "83.8%"],
    ["CRM-Prompt", "10.2 [7.8, 13.2]", "10.8", "10.0", "9.2", "19.8%", "83.5%"],
    ["CRM-Mask (Stop Interception)", "1.9 [1.0, 3.5]", "0.0", "5.0", "2.5", "1.9%", "91.5%"]
]
format_word_table("TABLE II\nOUR RESULTS ON THE 160 TEST TASKS (QWEN2.5-3B-INSTRUCT, OLLAMA CPU, M=8, 3 SEEDS, 480 TRIALS).", t2_headers, t2_rows)

# Table III
t3_headers = ["System", "Exact Match", "Lexical Perturbation", "Structural Inversion"]
t3_rows = [
    ["Vanilla", "5.0 [2.0, 12.2]", "10.0 [5.2, 18.5]", "8.8 [4.3, 17.0]"],
    ["Mem0-style RAG", "10.0 [5.2, 18.5]", "13.8 [7.9, 23.0]", "5.0 [2.0, 12.2]"],
    ["CRM-Prompt", "5.0 [2.0, 12.2]", "10.0 [5.2, 18.5]", "8.8 [4.3, 17.0]"],
    ["CRM-Mask", "1.2 [0.2, 6.7]", "5.0 [2.0, 12.2]", "0.0 [0.0, 4.6]"]
]
format_word_table("TABLE III\nFTA (%) WITH WILSON 95% CONFIDENCE INTERVALS BY PARAPHRASE TIER (80 TASKS, 1 SEED).", t3_headers, t3_rows)

# Table IV
t4_headers = ["System", "M = 4 (194 tokens)", "M = 12 (417 tokens)", "M = 30 (856 tokens)"]
t4_rows = [
    ["Vanilla", "13.3%", "13.3%", "13.3%"],
    ["Mem0-style RAG", "13.3%", "13.3%", "10.0%"],
    ["CRM-Prompt", "16.7%", "10.0%", "10.0%"],
    ["CRM-Mask", "6.7%", "6.7%", "6.7%"]
]
format_word_table("TABLE IV\nFTA (%) ACROSS DISTRACTOR TURNS M (30 TASKS, 1 SEED, MEASURED TOKENS).", t4_headers, t4_rows)

# Table V
t5_headers = ["τ", "Recall (%)", "Exact", "Lexical", "Struct", "FIR (%)", "F1 (%)"]
t5_rows = [
    ["0.15", "100.0", "100.0", "100.0", "100.0", "99.0", "85.8"],
    ["0.20", "100.0", "100.0", "100.0", "100.0", "61.0", "90.8"],
    ["0.25", "99.7", "100.0", "99.0", "100.0", "39.0", "93.7"],
    ["0.30", "99.3", "100.0", "99.0", "99.0", "10.0", "98.0"],
    ["0.35", "98.7", "99.0", "99.0", "98.0", "0.0", "99.3"],
    ["0.45", "93.7", "97.0", "92.0", "92.0", "0.0", "96.7"],
    ["0.55", "81.3", "86.0", "79.0", "79.0", "0.0", "89.7"],
    ["0.65", "57.7", "65.0", "60.0", "48.0", "0.0", "73.2"]
]
format_word_table("TABLE V\nROUTER SENSITIVITY ACROSS 300 PARAPHRASE PROBES AND 100 BENIGN QUERIES FOR VARYING τ.", t5_headers, t5_rows)

# Table VI
t6_headers = ["Cand.", "Phrasing Style", "Development FTA [95% CI]"]
t6_rows = [
    ["1", "Direct Natural", "50.0 [35.2, 64.8]"],
    ["2", "Action Imperative", "60.0 [44.6, 73.7]"],
    ["3", "Execution Rule (Used on Test)", "17.5 [8.7, 32.0]"],
    ["4", "Agent Role Instruction", "55.0 [39.8, 69.3]"],
    ["5", "Concise Direct (Dev Winner)", "62.5 [47.0, 75.8]"]
]
format_word_table("TABLE VI\nDIRECTIVE WORDING ABLATION ON THE 40 DEVELOPMENT TASKS (EVALUATED WITHOUT DISTRACTORS).", t6_headers, t6_rows)

# SECTION VI
add_heading_1("VI. RELATED WORK")
add_heading_2("A. Agent Memory and Episodic Retrieval Systems")
add_body("Contemporary LLM agent frameworks maintain conversational state using episodic context stores and natural language reflection [2], [3], [4]. Reflexion [4] and ReAct [5] prompt agents to critique execution faults and append verbal post-mortems into context. While effective within a short reasoning trajectory, self-reflective notes fade as interaction histories grow long.")

add_body("Retrieval-augmented memory systems such as Mem0 [2] and MemGPT [3] preserve long-term facts by indexing dialogue segments into external vector databases. Qin et al. [1] demonstrated through ImplicitMemBench that descriptive retrieval falters on negative constraints. Unstructured notes remain passive passages that struggle to overcome the established generation priors of the base model [8].")

add_heading_2("B. Constrained Decoding and Policy Steering")
add_body("Model alignment methods typically adjust parameter weights via reinforcement learning from human feedback or direct preference optimization [7]. Toolformer [6] fine-tunes models to generate tool calls autonomously. Offline updates provide strong default competencies but cannot accommodate dynamic runtime constraints such as transient service outages or rotated credentials.")

add_body("Inference-time steering modifies model behavior without changing underlying parameters [11], [12]. Constrained decoding approaches alter logit distributions to enforce formal syntax rules. CRM builds upon this principle by combining fast vector routing with targeted inhibitory token suppression.")

add_heading_2("C. Conditioned Inhibition Principles")
add_body("CRM draws conceptual inspiration from classical conditioning and the Rescorla-Wagner associative learning model [9], [10]. In physiological systems, conditioned inhibition allows organisms to suppress reflexive actions when exposed to learned warning stimuli. We map conditioned stimuli and environmental feedback onto routing tuples as an engineering analogy rather than a biological claim.")

# SECTION VII
add_heading_1("VII. LIMITATIONS")
add_body("Our empirical findings reflect specific experimental boundaries that should guide interpretation:\n"
         "1) Benchmark Re-Creation: The 200 tasks represent our independent re-creation of ImplicitMemBench. Published results in Table I originate from a distinct evaluation suite and serve solely as background reference.\n"
         "2) Model Scale: All experiments evaluated a single small model (Qwen2.5-3B-Instruct) on local CPU hardware. We cannot extrapolate these findings to larger parameter regimes.\n"
         "3) Truncated Ablations: Paraphrase and horizon runs evaluated 80 and 30 tasks respectively with a single seed. The M=120 condition did not finish due to CPU timeouts. CRM-Prompt mirrored Vanilla across paraphrase tiers, an effect requiring further investigation.\n"
         "4) Mask Implementation: Our mask variant relied on Ollama stop sequences and prefix truncation rather than direct logit manipulation, contributing to its low 1.9% FTA score. Evaluating true PyTorch logit processors remains open.\n"
         "5) Directive Selection: The directive phrasing used on the test set scored lowest during development tuning. The development winner was not evaluated on the test set, creating an avoidable performance penalty.\n"
         "6) Scoring Granularity: Evaluation relies on exact string matching against expected function identifiers. Correct resolutions expressed through alternative syntax are scored as errors.\n"
         "7) Benign Control Coverage: The benign control set contains 100 developer queries. False inhibition increases to 10.0% at tau = 0.30, indicating a narrow operating window. Testing on open-domain traffic remains necessary.")

# SECTION VIII
add_heading_1("VIII. CONCLUSION AND FUTURE WORK")
add_body("We presented Contrastive Reflexive Memory, an execution framework that converts tool failures into conditioned reflex rules and routes incoming instructions using a single matrix-vector product. The semantic router demonstrated high reliability: at tau = 0.35, it achieved 98.7% trigger recall with 0.0% false inhibition on benign queries, executing in tens of microseconds for the vector product and 17.2 milliseconds end-to-end on CPU hardware.")

add_body("The generation side encountered substantial bottlenecks. On a 3B model, prompt directives improved upon vanilla generation but trailed descriptive retrieval, and our stop-sequence mask variant proved ineffective. Future work will deploy true logit-level suppression via PyTorch transformers, evaluate the winning development directive on the full test benchmark across multiple seeds, expand the benign query suite, and examine behavior on 7B and 70B parameter models. Until such scaled evaluations are completed, the semantic router represents the verified contribution of this work, while reliable failure actuation on small models remains an open engineering challenge.")

# REFERENCES
add_heading_1("REFERENCES")
refs = [
    "[1] Y. Qin, S. Wang, X. Zhao, Y. Lin, Z. Liu, and M. Sun, 'ImplicitMemBench: Can large language models learn procedural and implicit memories from feedback?,' in Proc. 64th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2026.",
    "[2] P. Chhikara, T. Sharma, and D. Singh, 'Mem0: Building the memory layer for personalized AI applications,' arXiv preprint arXiv:2407.03224, 2024.",
    "[3] C. Packer, V. Fang, S. G. Patil, K. Lin, S. Wooders, and J. E. Gonzalez, 'MemGPT: Towards LLMs as operating systems,' arXiv preprint arXiv:2310.08560, 2023.",
    "[4] N. Shinn, F. Cassano, E. Labash, A. Gopinath, K. Narasimhan, and S. Yao, 'Reflexion: Language agents with verbal reinforcement learning,' in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, pp. 8634–8652, 2023.",
    "[5] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, 'ReAct: Synergizing reasoning and acting in language models,' in Int. Conf. Learn. Represent. (ICLR), 2023.",
    "[6] T. Schick, J. Dwivedi-Yu, R. Dessi, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom, 'Toolformer: Language models can teach themselves to use tools,' in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, pp. 68539–68551, 2023.",
    "[7] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, 'Direct preference optimization: Your language model is secretly a reward model,' in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 36, pp. 53728–53741, 2023.",
    "[8] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, 'Attention is all you need,' in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 30, 2017.",
    "[9] I. P. Pavlov, Conditioned Reflexes: An Investigation of the Physiological Activity of the Cerebral Cortex. London: Oxford University Press, 1927.",
    "[10] R. A. Rescorla and A. R. Wagner, 'A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement,' in Classical Conditioning II: Current Research and Theory, A. H. Black and W. F. Prokasy, Eds. New York: Appleton-Century-Crofts, 1972, pp. 64–99.",
    "[11] T. Brown et al., 'Language models are few-shot learners,' in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, pp. 1877–1901, 2020.",
    "[12] H. Touvron et al., 'LLaMA: Open and efficient foundation language models,' arXiv preprint arXiv:2302.13971, 2023.",
    "[13] L. Ouyang et al., 'Training language models to follow instructions with human feedback,' in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 35, pp. 27730–27744, 2022.",
    "[14] G. Mialon et al., 'Augmented language models: A survey,' arXiv preprint arXiv:2302.07842, 2023.",
    "[15] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, 'BERT: Pre-training of deep bidirectional transformers for language understanding,' in Proc. NAACL-HLT, pp. 4171–4186, 2019.",
    "[16] N. Reimers and I. Gurevych, 'Sentence-BERT: Sentence embeddings using Siamese BERT-networks,' in Proc. EMNLP-IJCNLP, pp. 3982–3992, 2019."
]

for ref in refs:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(ref)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(8.5)

out_docx_path = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\reflexive_inhibition_paper.docx'
doc.save(out_docx_path)
print(f"Pristine Word document saved successfully: {out_docx_path}")

# Copy to downloads as well
import shutil
shutil.copy(out_docx_path, r'c:\Users\Naman\Downloads\reflexive_inhibition_paper.docx')
print("Copied pristine Word document to Downloads folder!")
