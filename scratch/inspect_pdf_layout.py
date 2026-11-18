import fitz # PyMuPDF

doc = fitz.open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_conference.pdf')
for pno in range(len(doc)):
    page = doc[pno]
    print(f"=== PAGE {pno+1} ===")
    images = page.get_images()
    print(f"Images on page {pno+1}: {len(images)}")
    for img in images:
        xref = img[0]
        base_img = doc.extract_image(xref)
        print(f"   Image xref {xref}: {base_img['width']}x{base_img['height']}, {base_img['ext']}")
    text = page.get_text()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    print(f"Text lines ({len(lines)}):")
    for l in lines[:8]:
        print("   ", l)
    if len(lines) > 8:
        print("    ... and", len(lines) - 8, "more lines")
