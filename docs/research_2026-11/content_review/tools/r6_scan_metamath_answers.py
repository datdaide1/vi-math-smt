import json, re, sys, collections
# Precision check snippets for macro patterns + corrected median count + answer-line mismatch categorisation.
sys.stdout.reconfigure(encoding='utf-8')
S = sys.argv[1]
VNL = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
MACRO = re.compile(r'\\[a-zA-Z]*[' + VNL + VNL.upper() + r'][^\s{$]*|\\ (?:tương|song|chấm|góc|đóng)\S*')
BSSPACE = re.compile(r'\\ (?:geq|leq|cdot|frac|sqrt|times|circ|equiv|mathbf|neq|dots|pi)\b')
ex = json.load(open(S + '/work/mm_scan3_ex.json'))
want = {'r_macro_translated': set(ex['r_macro_translated'][:15]), 'any_bs_space_macro': set(ex['any_bs_space_macro'][:8])}
allw = set().union(*want.values())
C = collections.Counter(); macro_tokens = collections.Counter()
def ans(t, vi):
    m = re.search((r'(?:Câu trả lời là|Đáp án là|câu trả lời là|đáp án là)' if vi else r'The answer is') + r'\s*:\s*(.*)$', t.strip(), re.S)
    return m.group(1).strip() if m else None
def norm(a): return re.sub(r'\s+', '', a or '').rstrip('.')
R = 'E:/MATH-REASONING-THESIS/vi-math-smt/data/test/metaMATH/metaMathQA-395K_json_vi.jsonl'
with open(R, encoding='utf-8') as f:
    for i, l in enumerate(f):
        r = json.loads(l)
        qe, qv, re_, rv = r['query_en'], r['query_vi'], r['response_en'], r['response_vi']
        for m in MACRO.finditer(qv + ' ' + rv):
            macro_tokens[m.group()[:14]] += 1
        if re.search(r'\bmedian\b', qe, re.I):
            C['median_en'] += 1
            if not re.search(r'trung vị', qv, re.I):
                C['median_lost_ci'] += 1
        ae, av = ans(re_, False), ans(rv, True)
        if av is None:
            C['vi_no_answer_line'] += 1
        elif ae is not None:
            a, b = norm(ae), norm(av)
            if a != b:
                C['ans_mismatch'] += 1
                if a.replace('.', ',') == b or a == b.replace(',', '.'):
                    C['ans_mismatch_decimal_comma_only'] += 1
                elif re.sub(r'[\\{}]', '', a) == re.sub(r'[\\{}]', '', b):
                    C['ans_mismatch_latex_spacing_only'] += 1
                else:
                    C['ans_mismatch_other'] += 1
                    if C['ans_mismatch_other'] <= 25:
                        print('ANS-OTHER', i, repr(ae[:60]), '|', repr(av[:60]))
        if i in allw:
            for k, s in want.items():
                if i in s:
                    pat = MACRO if k == 'r_macro_translated' else BSSPACE
                    txt = qv + ' || ' + rv
                    ms = [txt[max(0, m.start() - 25):m.end() + 15] for m in pat.finditer(txt)][:2]
                    print('SNIP', k, i, ms)
print(json.dumps(dict(C), ensure_ascii=False))
print('TOP macro tokens:', macro_tokens.most_common(40))
