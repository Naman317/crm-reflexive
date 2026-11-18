import pypdfium2 as pdfium

pdf = pdfium.PdfDocument(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_author_compile.pdf')
page = pdf[0]
image = page.render(scale=2.0).to_pil()
image.save(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\test_author_page_1.png')
print("Saved test_author_page_1.png")
