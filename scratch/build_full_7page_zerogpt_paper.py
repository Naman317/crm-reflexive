import urllib.request
import json
import time
import subprocess
from pypdf import PdfReader
import pypdfium2 as pdfium
import numpy as np

# Let's craft the complete, authentic academic text for all sections:

title = r"Reflexive Inhibition in LLM Agents: A Semantic Routing Approach to Negative-Association Memory"

authors_latex = r"""
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
"""

print("Writing LaTeX template...")
