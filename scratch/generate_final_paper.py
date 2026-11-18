import os
import subprocess
import json
import urllib.request
import time
from pypdf import PdfReader
import pypdfium2 as pdfium
import numpy as np

tex_content = r"""\documentclass[10pt,conference]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage{times}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{url}
\usepackage{microtype}

% Hyphenation and spacing control
\hyphenpenalty=10000
\exhyphenpenalty=10000
\tolerance=3000
\emergencystretch=1.5em

% Float controls
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\dblfloatpagefraction}{0.85}

\setlength{\textfloatsep}{5pt plus 1pt minus 1pt}
\setlength{\floatsep}{5pt plus 1pt minus 1pt}
\setlength{\intextsep}{5pt plus 1pt minus 1pt}
\setlength{\dbltextfloatsep}{6pt plus 1pt minus 1pt}
\setlength{\dblfloatsep}{6pt plus 1pt minus 1pt}

\title{\textbf{\LARGE Reflexive Inhibition in LLM Agents: \\ A Semantic Routing Approach to Negative-Association Memory}}

\author{
\noindent\begin{minipage}[t]{0.32\textwidth}
\centering
{\normalsize \textbf{Ayush Raj Vishnu}}\\[0.2ex]
{\footnotesize \textit{Dept. of Computer Science \& Engineering}}\\[0.1ex]
{\footnotesize \textit{Lovely Professional University}}\\[0.1ex]
{\footnotesize Punjab, India}\\[0.1ex]
{\footnotesize Reg. No: 12619994}\\[0.1ex]
{\footnotesize \texttt{ayushrajvishnu0311@gmail.com}}
\end{minipage}\hfill
\begin{minipage}[t]{0.32\textwidth}
\centering
{\normalsize \textbf{Naman Sharma}}\\[0.2ex]
{\footnotesize \textit{Dept. of Computer Science \& Engineering}}\\[0.1ex]
{\footnotesize \textit{Lovely Professional University}}\\[0.1ex]
{\footnotesize Punjab, India}\\[0.1ex]
{\footnotesize Reg. No: 12610802}\\[0.1ex]
{\footnotesize \texttt{namansharma3194@gmail.com}}
\end{minipage}\hfill
\begin{minipage}[t]{0.32\textwidth}
\centering
{\normalsize \textbf{Vishnu Narayanan P}}\\[0.2ex]
{\footnotesize \textit{Dept. of Computer Science \& Engineering}}\\[0.1ex]
{\footnotesize \textit{Lovely Professional University}}\\[0.1ex]
{\footnotesize Punjab, India}\\[0.1ex]
{\footnotesize Reg. No: 12614204}\\[0.1ex]
{\footnotesize \texttt{222vishnunarayanan@gmail.com}}
\end{minipage}\\[1.5ex]
\begin{minipage}[t]{\textwidth}
\centering
{\normalsize \textbf{Dr. Baljit Singh Saini}}\\[0.2ex]
{\footnotesize \textit{Professor, School of Computing and AI, Lovely Professional University, Punjab, India}}\\[0.1ex]
{\footnotesize \texttt{baljit.22078@lpu.co.in}}
\end{minipage}
}

\begin{document}

\maketitle

\begin{abstract}
Large language models (LLMs) have been increasingly used to interact with external environments (e.g., compilers, databases, APIs) as goal-driven agents. However, it remains challenging for these language agents to quickly and efficiently learn from execution failures, as standard memory systems append historical conversational notes that compete with surrounding dialogue under multi-head self-attention. We propose Contrastive Reflexive Memory (CRM), a novel framework to protect language agents not by updating model weights, but instead through conditioned inhibitory reflexes. Concretely, CRM agents record execution failure signals, then maintain a compact buffer of conditioned reflex rules to induce immediate action suppression in subsequent trials. CRM is flexible enough to incorporate both prompt-level directives (CRM-Prompt) and decoding-time token masking (CRM-Mask), evaluating incoming queries through a single matrix-vector dot product against stored trigger embeddings. The dominant agent memory models are based on complex retrieval networks that append historical conversational notes into prompt context. The best performing models also connect external vector databases through dense embedding lookups. We propose a simple runtime architecture based solely on conditioned inhibitory reflexes, dispensing with conversational text retrieval entirely. On our test evaluation, the router achieves 98.7\% recall on 300 paraphrased trigger probes with 0.0\% false inhibition on benign queries at threshold 0.35, requiring 17.2 milliseconds per lookup on commodity CPU hardware. We also conduct extensive ablation studies across 160 benchmark tasks using Qwen2.5-3B-Instruct, examining the gap between semantic routing precision and small-model generation capabilities where CRM-Prompt reaches 10.2\% accuracy while retrieval reaches 19.4\%. We report these empirical outcomes without selective omissions, formalize the underlying routing dynamics, and identify architectural bottlenecks in small-model negative conditioning.
\end{abstract}

\begin{IEEEkeywords}
Language model agents, tool execution, negative conditioning, semantic routing, failure recovery, memory systems.
\end{IEEEkeywords}

\section{Introduction}

Recent works such as ReAct [5], SayCan, Toolformer [6], and Reflexion [4] have demonstrated the feasibility of autonomous decision-making agents built on top of a large language model core. These methods use language models to generate textual thoughts and actions that execute external API calls across operating systems and databases. When deployed in interactive environments, an agent must register execution faults and prevent their recurrence. A human software engineer who encounters a fatal permission exception or an endpoint deprecation notices the failure signal and switches tools immediately. Current language models struggle with such negative constraints. When conversations lengthen, agents frequently lose track of negative directives and repeat failed calls.

Software agents powered by large language models execute actions by calling external tools in sequence. An agent needs to learn from bad actions so that it does not repeat them later. However, current language models fail to maintain negative constraints over long conversations. In this work, we introduce Contrastive Reflexive Memory to solve this problem using inhibitory reflexes. We design two implementations: CRM-Prompt for black-box models and CRM-Mask for open-weight models. Our setup uses a small sentence transformer with 384 dimensions. When the score is greater than 0.35, the tool call is blocked. Vector stores work well for answering questions about previous conversations. But they perform poorly when an agent must avoid an action. Other approaches guide language model outputs at inference time without modifying model weights.

\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{fig1_asymmetry_gap.png}
\caption{First-try accuracy across negative (inhibition) and positive (preference) conditions. The four left bars represent our empirical runs on classical tasks with Qwen2.5-3B; the right groups present published results for frontier models [1] for reference.}
\label{fig:asymmetry_gap}
\end{figure}

Existing memory frameworks such as Mem0 [2] and MemGPT [3] excel at retrieving factual snippets from dialogue histories. However, they remain ill-suited for negative conditioning. Storing an execution failure as an unstructured narrative note simply adds another paragraph to the prompt context. Under dense prompt loading, negative directives compete directly with prior associations and task descriptions. Fig. 1 illustrates this structural gap across published frontier models: accuracy on negative inhibition tasks lags behind positive preference tasks by over 45 percentage points. Our local small-model runs reflect low overall completion rates across both conditions.

This paper describes Contrastive Reflexive Memory (CRM), an execution layer inspired by classical conditioning principles [9], [10]. Rather than retaining raw conversation text, CRM maintains a compact buffer of failure reflexes in local memory. Incoming user queries are screened through an efficient similarity calculation against stored failure rules. Each reflex rule $r_i$ is represented as the formal tuple in (1):
\begin{equation}
r_i = \left(\mathbf{e}_{\text{CS}, i}, \, \mathcal{P}_{\text{CS}, i}, \, a_{\text{inhibit}, i}, \, \mathcal{V}_{\text{inhibit}, i}, \, a_{\text{safe}, i}\right)
\end{equation}
where $\mathbf{e}_{\text{CS}, i}$ denotes the normalized trigger vector, $\mathcal{P}_{\text{CS}, i}$ is the trigger instruction text, $a_{\text{inhibit}, i}$ is the forbidden tool action, $\mathcal{V}_{\text{inhibit}, i}$ contains the forbidden token identifiers, and $a_{\text{safe}, i}$ specifies the verified fallback procedure.

We present two operational mechanisms for reflex actuation: CRM-Prompt for black-box model APIs and CRM-Mask for open-weight model architectures. User instructions are mapped into dense vectors using a lightweight 384-dimensional sentence encoder [16]. When the cosine similarity meets or exceeds 0.35, the reflex gate fires. CRM-Prompt prepends a concise steering directive directly before the query text. CRM-Mask suppresses forbidden tool tokens at decoding time. We evaluate the framework on a 200-task suite modeled on ImplicitMemBench [1]. Our primary findings are:
\begin{enumerate}
\item We formalize failure avoidance as a compact buffer of conditioned reflex rules evaluated via a single-step cosine router, supporting both prompt-level and logit-level intervention paths.
\item We characterize the routing boundary: at threshold $\tau = 0.35$, the router flags 98.7\% of 300 paraphrased trigger probes with 0.0\% false inhibition on 100 benign queries, consuming a median of 17.2 ms on a consumer CPU.
\item We provide an honest accounting of generation limitations. On a 3B model, prompt-level injection yields modest gains over vanilla generation (10.2\% vs. 3.3\%) while trailing descriptive retrieval (19.4\%), and our stop-sequence version of CRM-Mask produced 1.9\% first-try accuracy.
\end{enumerate}

\section{Problem Formulation}

This section details our benchmark protocol, scoring metrics, and failure modes observed in descriptive context memory.

\subsection{Implicit Memory Evaluation Protocol}

Following the experimental design of ImplicitMemBench [1], each evaluation episode proceeds through three successive phases: learning, distraction, and testing. The model produces tool actions autoregressively token by token according to the conditional distribution in (2):
\begin{equation}
P(y_k \mid y_{<k}, \, \mathcal{C}) = \text{softmax}\left(\mathbf{W}_{\text{head}} \mathbf{h}_k\right)
\end{equation}
where $y_k$ is the generated action token, $\mathcal{C}$ denotes the active prompt context, $\mathbf{h}_k$ is the hidden representation, and $\mathbf{W}_{\text{head}}$ is the language model head matrix.

The evaluation lifecycle is structured into three distinct stages:
\begin{itemize}
\item \textbf{Learning Phase ($\mathcal{T}_{\text{learn}}$):} The agent receives prompt $x_{\text{learn}}$, executes action $a_{\text{inhibit}}$, and encounters an environmental exception such as permission denied, a resource lock, or an HTTP 429 status code. The environment feedback provides the verified safe fallback action $a_{\text{safe}}$.
\item \textbf{Interference Phase ($\mathcal{T}_{\text{interfere}}$):} The conversation progresses through $M$ unrelated dialogue turns, pushing the failure episode out of the immediate attention focus.
\item \textbf{Testing Phase ($\mathcal{T}_{\text{test}}$):} The agent receives an evaluation prompt $x_{\text{test}}$, testing whether it maintains the negative constraint without repeating $a_{\text{inhibit}}$.
\end{itemize}

\subsection{First-Try Accuracy and the Asymmetry Deficit}

We evaluate agent responses using First-Try Accuracy (FTA) as defined in (3):
\begin{equation}
\text{FTA} = \frac{1}{N} \sum_{j=1}^N \mathbb{I}\left(a_{\text{inhibit}, j} \notin y_{\text{test}, j} \land a_{\text{safe}, j} \in y_{\text{test}, j}\right) \times 100\%
\end{equation}
where $N$ is the dataset size, $a_{\text{safe}, j}$ is the required substitute tool, $a_{\text{inhibit}, j}$ is the prohibited call, and $\mathbb{I}(\cdot)$ is the indicator function. The trial succeeds only if the response contains $a_{\text{safe}}$ and strictly avoids $a_{\text{inhibit}}$.

For positive preference scenarios, the model must produce the requested target. The asymmetry deficit $\Delta$ measures the performance gap between positive and negative tasks:
\begin{align}
\text{FTA}_{\text{pos}} &= \frac{1}{N} \sum_{j=1}^N \mathbb{I}\left(a_{\text{safe}, j} \in y_{\text{test}, j}\right) \times 100\% \\
\Delta &= \text{FTA}_{\text{pos}} - \text{FTA}_{\text{neg}}
\end{align}
In published benchmarks on frontier models, $\Delta$ exceeds 46.0\%, indicating a pronounced deficit in processing negative associations.

\subsection{Why Declarative Notes May Fail}

Dense vector indices retrieve historical facts effectively, but they process negative constraints as ordinary textual passages. When the test prompt is paraphrased, semantic similarity to the stored note can drop below retrieval thresholds. Even when retrieved successfully, the note sits far back in the prompt context. Under multi-head self-attention mechanisms, attention weights on distant historical tokens decay as dialogue grows [8].

Autoregressive language models are generative by design. Pretrained representations associate broad intent concepts with high-frequency training tokens. Suppressing an established prior requires active steering rather than passive background notes. CRM addresses this by replacing passage retrieval with direct semantic routing paired with immediate directive actuation.

\section{Contrastive Reflexive Memory Architecture}

CRM extracts failure events from execution signals and caches them in a local memory buffer. The architecture comprises three sequential modules: failure extraction, semantic routing, and inhibitory actuation.

\subsection{Stage 1: Conditioned Failure Extractor}

Drawing on classical conditioning principles [9], [10], CRM records failures as lightweight associations. When a tool call generates an error, the system monitors the environment failure signals in (6):
\begin{equation}
u_{\text{US}} \in \left\{\text{permission denied}, \text{deadlock}, \text{HTTP 429}, \text{syntax error}\right\}
\end{equation}
Upon encountering $u_{\text{US}}$, the extractor records the trigger prompt $\mathcal{P}_{\text{CS}, i}$, tags the prohibited tool call $a_{\text{inhibit}, i}$, and stores the safe alternative $a_{\text{safe}, i}$. The trigger text is projected into an L2-normalized embedding vector $\mathbf{e}_{\text{CS}, i}$ via (7):
\begin{equation}
\mathbf{e}_{\text{CS}, i} = \frac{f_{\text{emb}}(\mathcal{P}_{\text{CS}, i})}{\|f_{\text{emb}}(\mathcal{P}_{\text{CS}, i})\|_2} \in \mathbb{R}^D
\end{equation}
where $f_{\text{emb}}$ represents a 384-dimensional sentence transformer model (\texttt{all-MiniLM-L6-v2}) [16]. The resulting reflex rule $r_i$ is appended to the active reflex buffer $\mathcal{R}$.

\subsection{Stage 2: Paraphrase-Invariant Semantic Router}

The router screens each new user query against the entire buffer of reflex rules in a single pass:
\begin{enumerate}
\item \textbf{Query Vector Projection:} For incoming instruction $x_t$ at turn $t$, the router computes the normalized vector $\mathbf{e}_t = f_{\text{emb}}(x_t) / \|f_{\text{emb}}(x_t)\|_2$.
\item \textbf{Matrix-Vector Product:} The query vector is multiplied against the stacked buffer matrix $\mathbf{E}_{\mathcal{R}} \in \mathbb{R}^{K \times D}$ via (8):
\begin{equation}
\mathbf{s} = \mathbf{E}_{\mathcal{R}} \mathbf{e}_t \in \mathbb{R}^{|\mathcal{R}|}
\end{equation}
where $\mathbf{s}$ holds cosine similarity scores for all stored trigger rules.
\item \textbf{Winner Selection:} The highest scoring rule index $i^*$ is identified via the argmax operation in (9):
\begin{equation}
i^* = \arg\max_i s_i, \quad s_t = s_{i^*}
\end{equation}
\item \textbf{Threshold Gating:} The binary gating function $g(s_t)$ in (10) determines reflex activation:
\begin{equation}
g(s_t) = \begin{cases} 1, & \text{if } s_t \ge \tau \\ 0, & \text{if } s_t < \tau \end{cases}
\end{equation}
We set the operating threshold to $\tau = 0.35$. Section V-C provides empirical justification for this operating point.
\end{enumerate}

\subsection{Stage 3: Contrastive Inhibitory Actuator}

The actuator modifies the generation process through two alternative pathways depending on model access permissions:
\begin{itemize}
\item \textbf{CRM-Prompt (Black-Box APIs):} For systems without token probability access, the actuator prepends a steering directive directly before the query, as formulated in (11):
\begin{equation}
\tilde{x}_t = \left[\text{Execution Rule: Avoid } a_{\text{inhibit}, i^*}. \text{ Call } a_{\text{safe}, i^*} \text{ directly.}\right] \circ x_t
\end{equation}
Placing the directive at recency offset zero maximizes self-attention allocation during generation.
\item \textbf{CRM-Mask (Open-Weight Models):} Where decoding logits are accessible, the actuator subtracts a penalty $\beta$ from the logits of forbidden tokens, as shown in (12):
\begin{equation}
\tilde{z}_{k, v} = z_{k, v} - \beta \cdot g(s_t) \cdot \mathbb{I}\left(v \in \mathcal{V}_{\text{inhibit}, i^*}\right)
\end{equation}
where $z_{k, v}$ is the raw logit for vocabulary token $v$ at step $k$, and $\beta > 0$ denotes the suppression magnitude. In our local benchmark, this variant operated through Ollama stop sequences and token prefix interception.
\end{itemize}

\section{Experimental Setup}

This section describes our task suite, baseline configurations, and local inference execution environment.

\subsection{Task Suite}

We constructed an evaluation suite of 200 tasks modeled on the taxonomy of ImplicitMemBench [1]. The collection is an independent re-creation designed for local benchmarking and consists of four components:
\begin{itemize}
\item \textbf{Classical Conditioning (100 tasks):} Evaluates positive preference and negative inhibition across file permissions, network protocols, tool deprecations, and authentication endpoints.
\item \textbf{Procedural Memory (100 tasks):} Tests multi-step operational habits, including argument inversions, session tokens, and custom serialization formats.
\item \textbf{Paraphrase Probes (300 probes):} Evaluates trigger generalization across three tiers: Exact match, Lexical substitution, and Structural inversion.
\item \textbf{Benign Controls (100 prompts):} Unrelated developer queries that must not activate failure reflexes, measuring the False Inhibition Rate (FIR).
\end{itemize}
We isolated 40 tasks as a development set for directive selection. The remaining 160 tasks form the frozen test benchmark.

\subsection{Systems and Inference Setup}

We evaluate four execution configurations:
\begin{enumerate}
\item \textbf{Vanilla:} The base model receives conversational history without memory augmentation.
\item \textbf{Mem0-style Retrieval:} The model retrieves the top three relevant failure notes from an embedding index and prepends them to the context window [2].
\item \textbf{CRM-Prompt:} The semantic router activates a contrastive directive at recency offset zero when $s_t \ge 0.35$.
\item \textbf{CRM-Mask:} The router enforces stop-sequence truncation when prohibited tokens are generated.
\end{enumerate}

All evaluations were executed locally via Ollama on an Intel Core i5 laptop CPU with Qwen2.5-3B-Instruct. We used greedy decoding ($\text{temperature} = 0.0$, 128 token generation limit). Each evaluation trial incorporates 8 distractor turns, except in the horizon scaling analysis. Embeddings were generated with \texttt{all-MiniLM-L6-v2}. The main benchmark includes three random seeds (42, 137, 2026), totaling 480 trials per method.

\section{Results}

\subsection{Main Benchmark Results}

Table I presents published figures from Qin et al. [1] for reference. Table II summarizes our empirical results across the 160 test tasks.

\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{fig2_qualitative_trace.png}
\caption{Qualitative response trace from scenario classical\_017 (seed 137). CRM-Prompt successfully directs the model to emit the verified substitute action, whereas other baselines fail to complete the safe call.}
\label{fig:qualitative_trace}
\end{figure}

Unconditioned vanilla generation achieves 3.3\% FTA overall. Mem0-style retrieval reaches 19.4\%, the highest task success rate in the group. CRM-Prompt attains 10.2\% FTA. A paired McNemar test on matched task-seed pairs shows Mem0 significantly outperforming CRM-Prompt (67 unique wins for Mem0 versus 23 for CRM-Prompt, $p < 0.001$), while CRM-Prompt significantly outperforms Vanilla (40 wins versus 7, $p < 0.001$). The advantage of Mem0 stems primarily from procedural tasks (30.4\% vs. 10.8\%), where detailed context notes help small models structure multi-argument syntax.

Failure analysis reveals that most errors are omissions rather than violations. All systems avoid the forbidden action in over 76\% of trials, with CRM-Mask reaching 91.5\% avoidance. However, generating the verified safe call remains rare: CRM-Prompt emits $a_{\text{safe}}$ in 19.8\% of trials, while CRM-Mask does so in only 1.9\%. As illustrated in Fig. 2, the 3B model often outputs explanations, apologies, or conversational refusals instead of issuing the functional tool call.

\subsection{Paraphrase Robustness and Interference Horizon}

Table III presents accuracy across paraphrase tiers on 80 classical tasks with one seed. Performance remains low across all tiers, with overlapping confidence intervals. CRM-Prompt matches Vanilla across all three tiers (5.0\% Exact, 10.0\% Lexical, 8.8\% Structural), as illustrated in Fig. 3. Because exact-match baselines remain near 5\%, calculating retention ratios yields uninterpretable divergence, so we report raw per-tier accuracy.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\columnwidth]{fig3_paraphrase_decay.png}
\caption{First-try accuracy across syntactic paraphrase tiers on 80 classical tasks (one seed). Accuracies remain bounded between 0\% and 14\%.}
\label{fig:paraphrase_tiers}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\columnwidth]{fig4_tau_operating_curve.png}
\caption{Semantic router operating characteristic across routing thresholds $\tau$. Trigger recall is evaluated on 300 paraphrase probes; false inhibition rate is measured on 100 benign control prompts.}
\label{fig:tau_curve}
\end{figure}

Table IV reports performance across distractor turns $M \in \{4, 12, 30\}$ on 30 tasks with one seed. Prompt lengths increase from 194 to 856 tokens. Model accuracy remains stable across horizons, reflecting baseline variance on 30 tasks. The $M=120$ condition timed out on our laptop CPU setup, completing only 3 trials per system, and is omitted from reported tables.

\subsection{Router Sensitivity and False Inhibition}

The semantic router demonstrates robust operating characteristics. We swept threshold $\tau \in [0.15, 0.65]$ across 300 paraphrased trigger probes and 100 benign developer queries (Table V and Fig. 4). At low thresholds ($\tau = 0.15$), the router triggers indiscriminately, causing a 99.0\% false inhibition rate. At $\tau = 0.35$, trigger recall reaches 98.7\% (99.0\% exact, 99.0\% lexical, 98.0\% structural) with 0.0\% false inhibition on benign prompts, achieving 100.0\% precision and an F1 score of 99.3\%.

Routing latency remains negligible. Across 1,000 lookups against 100 stored rules on a laptop CPU, full routing completed in a median of 17.2 ms (95th percentile: 26.2 ms). Over 99\% of this time corresponds to embedding generation (median 17.16 ms). The isolated matrix-vector dot product consumed a median of 16.55 microseconds.

\begin{table*}[t]
\centering
\caption{Published First-Try Accuracy (FTA \%) from Qin et al. [1], shown for reference only. Our suite is an independent re-creation.}
\label{tab:published_results}
\small
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
Model & Procedural Memory & Classical (All) & Inhibition (Neg.) & Preference (Pos.) & Asymmetry Deficit $\Delta$ \\
\midrule
DeepSeek-R1 & 58.0 & 69.7 & 42.0 & 88.0 & 46.0 \\
Qwen3-32B & 55.0 & 67.0 & 40.5 & 86.5 & 46.0 \\
GPT-5 (Preview) & 52.0 & 64.0 & 38.0 & 85.0 & 47.0 \\
Qwen3-8B & 51.0 & 64.0 & 37.0 & 84.0 & 47.0 \\
Claude-4.1-Opus & 35.0 & 41.7 & 22.0 & 76.0 & 54.0 \\
LLaMA-3.1-8B-Instruct & 32.0 & 38.3 & 19.5 & 74.0 & 54.5 \\
Qwen2.5-7B-Instruct & 28.0 & 35.7 & 17.6 & 75.0 & 57.4 \\
\bottomrule
\end{tabular*}
\end{table*}

\begin{table*}[t]
\centering
\caption{Our results on the 160 test tasks (Qwen2.5-3B-Instruct, Ollama CPU, $M=8$, 3 seeds, 480 trials per system). FTA is reported with Wilson 95\% confidence intervals. The final two columns report the proportion of trials emitting $a_{\text{safe}}$ and avoiding $a_{\text{inhibit}}$.}
\label{tab:main_results}
\small
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}
\toprule
System & FTA Overall [95\% CI] & Procedural FTA & Classical Neg. & Classical Pos. & Safe Call Emitted (\%) & Forbidden Avoided (\%) \\
\midrule
Vanilla & 3.3 [2.1, 5.3] & 0.0 & 8.3 & 5.0 & 7.3 & 76.2 \\
Mem0-style Retrieval & 19.4 [16.1, 23.1] & 30.4 & 13.3 & 3.3 & 24.4 & 83.8 \\
CRM-Prompt & 10.2 [7.8, 13.2] & 10.8 & 10.0 & 9.2 & 19.8 & 83.5 \\
CRM-Mask (Stop Interception) & 1.9 [1.0, 3.5] & 0.0 & 5.0 & 2.5 & 1.9 & 91.5 \\
\bottomrule
\end{tabular*}
\end{table*}

\begin{table}[htbp]
\centering
\caption{FTA (\%) with Wilson 95\% confidence intervals by paraphrase tier (80 classical tasks, one seed).}
\label{tab:paraphrase_results}
\small
\begin{tabularx}{\columnwidth}{lXXX}
\toprule
System & Exact Match & Lexical Pert. & Structural Inv. \\
\midrule
Vanilla & 5.0 [2.0, 12.2] & 10.0 [5.2, 18.5] & 8.8 [4.3, 17.0] \\
Mem0-style RAG & 10.0 [5.2, 18.5] & 13.8 [7.9, 23.0] & 5.0 [2.0, 12.2] \\
CRM-Prompt & 5.0 [2.0, 12.2] & 10.0 [5.2, 18.5] & 8.8 [4.3, 17.0] \\
CRM-Mask & 1.2 [0.2, 6.7] & 5.0 [2.0, 12.2] & 0.0 [0.0, 4.6] \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[htbp]
\centering
\caption{FTA (\%) across distractor turns $M$ (30 classical tasks, one seed, measured prompt tokens in top row).}
\label{tab:horizon_results}
\small
\begin{tabularx}{\columnwidth}{lXXX}
\toprule
System & $M = 4$ & $M = 12$ & $M = 30$ \\
\midrule
Prompt Tokens & 194 tokens & 417 tokens & 856 tokens \\
Vanilla & 13.3\% & 13.3\% & 13.3\% \\
Mem0-style RAG & 13.3\% & 13.3\% & 10.0\% \\
CRM-Prompt & 16.7\% & 10.0\% & 10.0\% \\
CRM-Mask & 6.7\% & 6.7\% & 6.7\% \\
\bottomrule
\end{tabularx}
\end{table}

\subsection{Directive Wording on the Development Set}

We evaluated five directive phrasing styles across the 40 development tasks without distractor turns (Table VI). The resulting confidence intervals overlap substantially. Candidate 3 (``Execution Rule: Avoid X. Call Y directly.'') scored the lowest at 17.5\% FTA. This phrasing was inadvertently frozen for the main test evaluation. The development winner was Candidate 5 (62.5\% FTA). Consequently, test results for CRM-Prompt reflect a suboptimal directive and likely underestimate the potential of prompt-based intervention.

\begin{table}[htbp]
\centering
\caption{Directive wording ablation on the 40 development tasks (evaluated without distractor turns).}
\label{tab:dev_wording}
\small
\begin{tabularx}{\columnwidth}{clX}
\toprule
Cand. & Phrasing Style & Development FTA [95\% CI] \\
\midrule
1 & Direct Natural & 50.0 [35.2, 64.8] \\
2 & Action Imperative & 60.0 [44.6, 73.7] \\
3 & Execution Rule (Used on Test) & 17.5 [8.7, 32.0] \\
4 & Agent Role Instruction & 55.0 [39.8, 69.3] \\
5 & Concise Direct (Dev Winner) & 62.5 [47.0, 75.8] \\
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[htbp]
\centering
\caption{Router sensitivity across 300 paraphrase probes and 100 benign developer queries for varying thresholds $\tau$.}
\label{tab:router_sweep}
\small
\begin{tabularx}{\columnwidth}{lXXXXXX}
\toprule
$\tau$ & Recall & Exact & Lexical & Struct & FIR & F1 \\
\midrule
0.15 & 100.0 & 100.0 & 100.0 & 100.0 & 99.0 & 85.8 \\
0.20 & 100.0 & 100.0 & 100.0 & 100.0 & 61.0 & 90.8 \\
0.25 & 99.7 & 100.0 & 99.0 & 100.0 & 39.0 & 93.7 \\
0.30 & 99.3 & 100.0 & 99.0 & 99.0 & 10.0 & 98.0 \\
\textbf{0.35} & \textbf{98.7} & \textbf{99.0} & \textbf{99.0} & \textbf{98.0} & \textbf{0.0} & \textbf{99.3} \\
0.45 & 93.7 & 97.0 & 92.0 & 92.0 & 0.0 & 96.7 \\
0.55 & 81.3 & 86.0 & 79.0 & 79.0 & 0.0 & 89.7 \\
0.65 & 57.7 & 65.0 & 60.0 & 48.0 & 0.0 & 73.2 \\
\bottomrule
\end{tabularx}
\end{table}

\section{Related Work}

\subsection{Agent Memory and Episodic Retrieval Systems}

Contemporary LLM agent frameworks maintain conversational state using episodic context stores and natural language reflection [2], [3], [4]. Reflexion [4] and ReAct [5] prompt agents to critique execution faults and append verbal post-mortems into context. While effective within a short reasoning trajectory, self-reflective notes fade as interaction histories grow long.

Retrieval-augmented memory systems such as Mem0 [2] and MemGPT [3] preserve long-term facts by indexing dialogue segments into external vector databases. Qin et al. [1] demonstrated through ImplicitMemBench that descriptive retrieval falters on negative constraints. Unstructured notes remain passive passages that struggle to overcome the established generation priors of the base model [8].

\subsection{Constrained Decoding and Policy Steering}

Model alignment methods typically adjust parameter weights via reinforcement learning from human feedback or direct preference optimization [7]. Toolformer [6] fine-tunes models to generate tool calls autonomously. Offline updates provide strong default competencies but cannot accommodate dynamic runtime constraints such as transient service outages or rotated credentials.

Inference-time steering modifies model behavior without changing underlying parameters [11], [12]. Constrained decoding approaches alter logit distributions to enforce formal syntax rules. CRM builds upon this principle by combining fast vector routing with targeted inhibitory token suppression.

\subsection{Biologically Inspired Conditioned Inhibition}

CRM draws conceptual inspiration from Pavlovian conditioning and the Rescorla-Wagner associative learning model [9], [10]. In physiological systems, conditioned inhibition allows organisms to suppress reflexive actions when exposed to learned warning stimuli. We map conditioned stimuli ($\mathcal{P}_{\text{CS}}$) and environmental feedback ($u_{\text{US}}$) onto routing tuples. This formulation serves as an engineering analogy rather than a biological claim.

\section{Limitations}

Our empirical findings reflect specific experimental boundaries that should guide interpretation:
\begin{enumerate}
\item \textbf{Benchmark Re-Creation:} The 200 tasks represent our independent re-creation of ImplicitMemBench. Published results in Table I originate from a distinct evaluation suite and serve solely as background reference.
\item \textbf{Model Scale:} All experiments evaluated a single small model (Qwen2.5-3B-Instruct) on local CPU hardware. We cannot extrapolate these findings to larger parameter regimes.
\item \textbf{Truncated Ablations:} Paraphrase and horizon runs evaluated 80 and 30 tasks respectively with a single seed. The $M=120$ condition did not finish due to CPU timeouts. CRM-Prompt mirrored Vanilla across paraphrase tiers, an effect requiring further investigation.
\item \textbf{Mask Implementation:} Our mask variant relied on Ollama stop sequences and prefix truncation rather than direct logit manipulation, contributing to its low 1.9\% FTA score. Evaluating true PyTorch logit processors remains open.
\item \textbf{Directive Selection:} The directive phrasing used on the test set scored lowest during development tuning. The development winner was not evaluated on the test set, creating an avoidable performance penalty.
\item \textbf{Scoring Granularity:} Evaluation relies on exact string matching against expected function identifiers. Correct resolutions expressed through alternative syntax are scored as errors.
\item \textbf{Benign Control Coverage:} The benign control set contains 100 developer queries. False inhibition increases to 10.0\% at $\tau = 0.30$, indicating a narrow operating window. Testing on open-domain traffic remains necessary.
\end{enumerate}

\section{Conclusion and Future Work}

We presented Contrastive Reflexive Memory, an execution framework that converts tool failures into conditioned reflex rules and routes incoming instructions using a single matrix-vector product. The semantic router demonstrated high reliability: at $\tau = 0.35$, it achieved 98.7\% trigger recall with 0.0\% false inhibition on benign queries, executing in tens of microseconds for the vector product and 17.2 milliseconds end-to-end on CPU hardware.

The generation side encountered substantial bottlenecks. On a 3B model, prompt directives improved upon vanilla generation but trailed descriptive retrieval, and our stop-sequence mask variant proved ineffective. Future work will deploy true logit-level suppression via PyTorch transformers, evaluate the winning development directive on the full test benchmark across multiple seeds, expand the benign query suite, and examine behavior on 7B and 70B parameter models. Until such scaled evaluations are completed, the semantic router represents the verified contribution of this work, while reliable failure actuation on small models remains an open engineering challenge.

\section*{References}

\begin{enumerate}
\itemsep=0.35ex
\item[{[1]}] Y. Qin, S. Wang, X. Zhao, Y. Lin, Z. Liu, and M. Sun, ``ImplicitMemBench: Can large language models learn procedural and implicit memories from feedback?,'' in \textit{Proc. 64th Annu. Meeting Assoc. Comput. Linguistics (ACL)}, 2026.
\item[{[2]}] P. Chhikara, T. Sharma, and D. Singh, ``Mem0: Building the memory layer for personalized AI applications,'' \textit{arXiv preprint arXiv:2407.03224}, 2024.
\item[{[3]}] C. Packer, V. Fang, S. G. Patil, K. Lin, S. Wooders, and J. E. Gonzalez, ``MemGPT: Towards LLMs as operating systems,'' \textit{arXiv preprint arXiv:2310.08560}, 2023.
\item[{[4]}] N. Shinn, F. Cassano, E. Labash, A. Gopinath, K. Narasimhan, and S. Yao, ``Reflexion: Language agents with verbal reinforcement learning,'' in \textit{Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol. 36, pp. 8634--8652, 2023.
\item[{[5]}] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, ``ReAct: Synergizing reasoning and acting in language models,'' in \textit{Int. Conf. Learn. Represent. (ICLR)}, 2023.
\item[{[6]}] T. Schick, J. Dwivedi-Yu, R. Dessi, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom, ``Toolformer: Language models can teach themselves to use tools,'' in \textit{Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol. 36, pp. 68539--68551, 2023.
\item[{[7]}] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, ``Direct preference optimization: Your language model is secretly a reward model,'' in \textit{Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol. 36, pp. 53728--53741, 2023.
\item[{[8]}] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, ``Attention is all you need,'' in \textit{Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol. 30, 2017.
\item[{[9]}] I. P. Pavlov, \textit{Conditioned Reflexes: An Investigation of the Physiological Activity of the Cerebral Cortex}. London: Oxford University Press, 1927.
\item[{[10]}] R. A. Rescorla and A. R. Wagner, ``A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement,'' in \textit{Classical Conditioning II: Current Research and Theory}, A. H. Black and W. F. Prokasy, Eds. New York: Appleton-Century-Crofts, 1972, pp. 64--99.
\item[{[11]}] T. Brown et al., ``Language models are few-shot learners,'' in \textit{Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol. 33, pp. 1877--1901, 2020.
\item[{[12]}] H. Touvron et al., ``LLaMA: Open and efficient foundation language models,'' \textit{arXiv preprint arXiv:2302.13971}, 2023.
\item[{[13]}] L. Ouyang et al., ``Training language models to follow instructions with human feedback,'' in \textit{Adv. Neural Inf. Process. Syst. (NeurIPS)}, vol. 35, pp. 27730--27744, 2022.
\item[{[14]}] G. Mialon et al., ``Augmented language models: A survey,'' \textit{arXiv preprint arXiv:2302.07842}, 2023.
\item[{[15]}] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, ``BERT: Pre-training of deep bidirectional transformers for language understanding,'' in \textit{Proc. NAACL-HLT}, pp. 4171--4186, 2019.
\item[{[16]}] N. Reimers and I. Gurevych, ``Sentence-BERT: Sentence embeddings using Siamese BERT-networks,'' in \textit{Proc. EMNLP-IJCNLP}, pp. 3982--3992, 2019.
\end{enumerate}

\end{document}
"""

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\reflexive_inhibition_paper.tex', 'w', encoding='utf-8') as f:
    f.write(tex_content)

print("Compiling LaTeX to PDF via tectonic...")
cmd = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\bin\tectonic.exe c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\reflexive_inhibition_paper.tex'
subprocess.run(cmd, shell=True, check=True)

pdf_path = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\reflexive_inhibition_paper.pdf'
doc = pdfium.PdfDocument(pdf_path)
print(f"Total pages: {len(doc)}")

reader = PdfReader(pdf_path)
for i, page in enumerate(reader.pages):
    txt = page.extract_text()
    print(f"Page {i+1}: words = {len(txt.split())}, characters = {len(txt)}")
