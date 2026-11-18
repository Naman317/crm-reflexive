from pypdf import PdfReader

t1 = "\n\n".join([p.extract_text() for p in PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_minipages_author.pdf').pages if p.extract_text()])
t2 = "\n\n".join([p.extract_text() for p in PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_conference.pdf').pages if p.extract_text()])

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\txt_minipages.txt', 'w', encoding='utf-8') as f:
    f.write(t1)

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\txt_conf.txt', 'w', encoding='utf-8') as f:
    f.write(t2)

print("Minipages length:", len(t1), "words:", len(t1.split()))
print("Conference length:", len(t2), "words:", len(t2.split()))
