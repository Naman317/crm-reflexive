import zipfile
import os

zip_path = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\crm_evaluation_results.zip'

with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as z:
    paper_dir = r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper'
    for root, dirs, files in os.walk(paper_dir):
        for f in files:
            full_path = os.path.join(root, f)
            arcname = os.path.relpath(full_path, r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition')
            z.write(full_path, arcname)
            print(f"Added to zip: {arcname}")

print("Zip package updated successfully!")
