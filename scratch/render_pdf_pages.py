import fitz # PyMuPDF
import sys

doc = fitz.open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_conference.pdf')
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    out_path = rf'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\page_{i+1}.png'
    pix.save(out_path)
    print(f"Saved {out_path}")
