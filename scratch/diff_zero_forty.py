from pypdf import PdfReader
import difflib

p_zero = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_minipages_author.pdf'
p_forty = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_complete_paper_png_headings.pdf'

t_zero = '\n\n'.join([p.extract_text() for p in PdfReader(p_zero).pages if p.extract_text()])
t_forty = '\n\n'.join([p.extract_text() for p in PdfReader(p_forty).pages if p.extract_text()])

print("Zero words:", len(t_zero.split()), "Forty words:", len(t_forty.split()))
print("\n--- DIFF ---")
for line in difflib.unified_diff(t_zero.splitlines(), t_forty.splitlines(), lineterm=''):
    print(line)
