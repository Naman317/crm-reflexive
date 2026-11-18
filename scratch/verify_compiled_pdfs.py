import pypdf, re, sys
sys.path.append(r'C:\Users\Naman\.gemini\config\skills\humanize-paper\scripts')
import turnitin_inspect

files = [
    r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_blind.pdf',
    r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_double_column_conference.pdf',
    r'c:\Users\Naman\Desktop\gen\crm-reflexive-inhibition\paper\ieee_single_column_journal.pdf'
]

for fpath in files:
    name = fpath.split('\\')[-1]
    print('=' * 70)
    print('AUDITING PDF:', name)
    reader = pypdf.PdfReader(fpath)
    print('Total pages:', len(reader.pages))
    
    full_text = ''
    hyphen_count = 0
    for p_idx, page in enumerate(reader.pages):
        txt = page.extract_text()
        full_text += txt + '\n'
        # Check for line-end hyphens like word-\n
        hyphens = re.findall(r'[a-zA-Z]+-\s*\n\s*[a-zA-Z]+', txt)
        hyphen_count += len(hyphens)
        if hyphens:
            print(f'  Page {p_idx+1} hyphens:', hyphens[:3])

    print(f'Total extracted chars: {len(full_text)}')
    print(f'Line-end hyphenated words count: {hyphen_count}')
    
    # Check if any vector chart tokens leaked
    chart_leaks = [w for w in ['Vanilla (Ollama)', '=0.0%', '=14.0%', '=57.4%', 'Exact Lexical Structural'] if w in full_text]
    print(f'Chart text leak detected: {chart_leaks}')

    # Extract sentences and test AI score
    sentences = [s.strip() for s in re.split(r'\. |\.\n', full_text) if len(s.strip().split()) >= 4]
    res = turnitin_inspect.audit_ai_writing(sentences)
    ai_pct = res['overall_ai_percent']
    flagged = f"{res['flagged_sentence_count']} / {res['total_sentences']}"
    print(f'Estimated AI Writing Score: {ai_pct:>4.1f}%')
    print(f'Flagged Sentences: {flagged}')
    if res['flagged_sentence_count'] > 0:
        for s in res['sentences']:
            if s['is_flagged']:
                print('  Flagged:', s['text'][:100])
                print('  Reasons:', s['reasons'])
print('=' * 70)
