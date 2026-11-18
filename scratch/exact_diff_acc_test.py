import difflib

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\accepted_complete_paper.txt', 'r', encoding='utf-8') as f:
    t_acc = f.read()

import urllib.request, json
with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\t1_zero.txt', 'r', encoding='utf-8') as f:
    t_pdf = f.read()

header_condensed = """Contrastive Reflexive Memory: Overcoming Negative-Association Blindness in Language Model Agents
Ayush Raj Vishnu, Naman Sharma, Sharuti, and Research Advisor
Dept. of Computer Science & Engineering, Lovely Professional University, Punjab, India
Reg. No: 12619994, 12610802, 12613054"""

abstract_pos = t_pdf.find("Abstract—")
body_and_abstract = t_pdf[abstract_pos:]
test_text = header_condensed + "\n\n" + body_and_abstract

print("DIFF:")
for line in difflib.unified_diff(t_acc.splitlines(), test_text.splitlines(), lineterm=''):
    print(line)
