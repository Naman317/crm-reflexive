with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\accepted_complete_paper.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("Paragraphs in accepted_complete_paper.txt:", len(text.split('\n\n')))
for i, p in enumerate(text.split('\n\n')):
    print(f"--- P{i+1} ({len(p.split())} words) ---")
    print(p[:120])
