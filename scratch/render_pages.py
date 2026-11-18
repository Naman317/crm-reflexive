import pypdfium2 as pdfium

pdf = pdfium.PdfDocument(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_conference.pdf')
for i, page in enumerate(pdf):
    image = page.render(scale=2.0).to_pil()
    out = rf'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\conf_page_{i+1}.png'
    image.save(out)
    print(f"Rendered {out}")
