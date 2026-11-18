from pypdf import PdfReader

reader = PdfReader(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\reflexive_inhibition_paper.pdf')
print("=== FINAL PDF AUDIT ===")
print("Page count:", len(reader.pages))
for i, page in enumerate(reader.pages):
    txt = page.extract_text()
    lines = [l.strip() for l in txt.split('\n') if l.strip()]
    first_line = lines[0] if lines else ''
    last_line = lines[-1] if lines else ''
    print(f"Page {i+1}: {len(txt.split()):>4} words | Starts: {first_line[:45]:<45} | Ends: {last_line[-45:]}")
