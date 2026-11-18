with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\txt_minipages.txt', encoding='utf-8') as f:
    t1 = f.read()

with open(r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\scratch\txt_conf.txt', encoding='utf-8') as f:
    t2 = f.read()

# Let's see what extra text is in t2:
diff = t2.replace(t1, "")
print("Extra text in t2:")
print(diff)
