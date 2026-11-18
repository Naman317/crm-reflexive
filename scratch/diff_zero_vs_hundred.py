from pypdf import PdfReader
import difflib

p1 = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_conference.pdf'
p2 = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_layout_balanced.pdf'

t1 = "\n\n".join([p.extract_text() or '' for p in PdfReader(p1).pages])
t2 = "\n\n".join([p.extract_text() or '' for p in PdfReader(p2).pages])

print("Length 1 (0.0% AI):", len(t1), "chars, words:", len(t1.split()))
print("Length 2 (100% AI):", len(t2), "chars, words:", len(t2.split()))

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\t1_zero.txt', 'w', encoding='utf-8') as f:
    f.write(t1)

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\t2_hundred.txt', 'w', encoding='utf-8') as f:
    f.write(t2)

diff = difflib.unified_diff(t1.splitlines(), t2.splitlines(), lineterm='')
print("\n--- DIFF ---")
for line in list(diff)[:30]:
    print(line)
