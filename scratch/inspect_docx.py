import docx

doc = docx.Document(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\reflexive_inhibition_paper.docx')
print("Num paragraphs:", len(doc.paragraphs))
print("Num tables:", len(doc.tables))
print("Num inline shapes (images):", len(doc.inline_shapes))
for i, table in enumerate(doc.tables):
    rows = len(table.rows)
    cols = len(table.columns)
    first_cell = table.cell(0, 0).text.strip().replace('\n', ' ')
    print(f"Table {i+1}: {rows}x{cols} | First cell: \"{first_cell[:40]}\"")

print("\nFirst 10 paragraphs:")
for i, p in enumerate(doc.paragraphs[:10]):
    if p.text.strip():
        print(f"P{i+1}: {p.text.strip()[:80]}")
