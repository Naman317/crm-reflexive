from pypdf import PdfReader

reader = PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_sections_compile.pdf')
txt = "\n\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
print(txt)
