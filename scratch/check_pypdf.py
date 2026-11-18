import pypdf

reader = pypdf.PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_conference.pdf')
t = reader.pages[0].extract_text()
print('Is "T wo" in page 1:', 'T wo' in t)
print('Is "Two" in page 1:', 'Two' in t)
print('Is "First-T ry" in page 1:', 'First-T ry' in t)
print('Is "First-Try" in page 1:', 'First-Try' in t)
for line in t.split('\n')[:10]:
    print('  ', repr(line))
