with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\accepted_complete_paper.txt', 'r', encoding='utf-8') as f:
    t_acc = f.read()

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\t1_zero.txt', 'r', encoding='utf-8') as f:
    t_pdf = f.read()

print("=== ACCEPTED TEXT (2.5% AI) ===")
print("Length:", len(t_acc), "Words:", len(t_acc.split()))
print(t_acc[:400])
print("\n=== PDF EXTRACTED TEXT (70.0% AI) ===")
print("Length:", len(t_pdf), "Words:", len(t_pdf.split()))
print(t_pdf[:400])
