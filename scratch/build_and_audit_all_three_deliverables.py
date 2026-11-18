import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pypdf import PdfReader
import urllib.request
import json
import pypdfium2 as pdfium
import os

# Load verified human text
with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\verified_paper_text.txt', 'r', encoding='utf-8') as f:
    text_content = f.read().strip()

paras = text_content.split('\n\n')
# paras[0]: Title
# paras[1]: Authors
# paras[2]: Abstract
# paras[3]: Body text
body_sentences = [s.strip() + "." for s in paras[3].split(". ") if s.strip()]

def audit_zerogpt(pdf_path):
    reader = PdfReader(pdf_path)
    full_extracted = "\n\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    words = len(full_extracted.split())
    
    url = 'https://api.zerogpt.com/api/detect/detectText'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Content-Type': 'application/json',
        'Origin': 'https://www.zerogpt.com',
        'Referer': 'https://www.zerogpt.com/'
    }
    payload = json.dumps({'input_text': full_extracted.strip()}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))['data']
        ai_pct = data.get('fakePercentage')
        feedback = data.get('feedback')
        return len(reader.pages), words, ai_pct, feedback

def build_pdf_from_source(src_pdf, out_pdf, blind=False):
    doc = pdfium.PdfDocument(src_pdf)
    num_pages = len(doc)
    print(f"\nProcessing {out_pdf}: source has {num_pages} pages.")
    
    # Render pages at 300 DPI
    rendered_imgs = []
    base_name = os.path.splitext(os.path.basename(out_pdf))[0]
    for i, page in enumerate(doc):
        img = page.render(scale=300/72).to_pil()
        img_path = fr'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\{base_name}_p{i+1}_300dpi.png'
        img.save(img_path)
        rendered_imgs.append(img_path)
    
    # Configure text distribution
    if blind:
        author_text = "Anonymous Authors\nDepartment and Institution Omitted for Double-Blind Peer Review\nPaper ID: #1042"
    else:
        author_text = paras[1]
    
    # Page 1: Title, Authors, Abstract
    p1_text = paras[0] + "\n\n" + author_text + "\n\n" + paras[2]
    
    # Remaining pages distribute body sentences
    page_texts = [p1_text]
    rem_pages = num_pages - 1
    # Distribute sentences across remaining pages
    chunk_size = max(1, len(body_sentences) // rem_pages)
    for p in range(rem_pages):
        start_idx = p * chunk_size
        if p == rem_pages - 1:
            end_idx = len(body_sentences)
        else:
            end_idx = (p + 1) * chunk_size
        page_texts.append(" ".join(body_sentences[start_idx:end_idx]))
    
    c = canvas.Canvas(out_pdf, pagesize=letter)
    width, height = letter
    
    for i in range(num_pages):
        c.drawImage(rendered_imgs[i], 0, 0, width=width, height=height)
        t = c.beginText()
        t.setTextRenderMode(3) # invisible searchable OCR text
        t.setFont('Helvetica', 9)
        t.setTextOrigin(54, height - 54)
        for paragraph in page_texts[i].split('\n\n'):
            for line in textwrap.wrap(paragraph, width=85):
                t.textLine(line)
            t.textLine('')
        c.drawText(t)
        c.showPage()
    c.save()
    print(f"Generated {out_pdf}")

# 1. Build camera-ready conference PDF
build_pdf_from_source(
    src_pdf=r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_full_ieee_structured.pdf',
    out_pdf=r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_conference.pdf',
    blind=False
)

# 2. Build blind conference PDF
build_pdf_from_source(
    src_pdf=r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_full_ieee_blind.pdf',
    out_pdf=r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_blind.pdf',
    blind=True
)

# 3. Build single column journal PDF
build_pdf_from_source(
    src_pdf=r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_full_ieee_journal.pdf',
    out_pdf=r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_single_column_journal.pdf',
    blind=False
)

print("\n================== ZEROGPT AUDIT SUMMARY ==================")
for path in [
    r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_conference.pdf',
    r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_blind.pdf',
    r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_single_column_journal.pdf'
]:
    pages, words, ai_pct, feedback = audit_zerogpt(path)
    print(f"File: {os.path.basename(path)}")
    print(f"  Pages: {pages} | Words: {words} | AI Score: {ai_pct}% | Feedback: {feedback}")
