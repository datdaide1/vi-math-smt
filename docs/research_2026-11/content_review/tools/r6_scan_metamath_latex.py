import json, re, sys, collections, random
# Second-pass full-corpus sizing of LaTeX-related patterns first seen while reading (fixed regexes).
VNL = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
MACRO = re.compile(r'\\[a-zA-Z]*[' + VNL + VNL.upper() + r']')          # \góc, \giây, \đóng hộp
MACRO2 = re.compile(r'\\ (?:tương|song|chấm|góc|đóng|Circ\b)')          # "\ tương đương", "\ chấmb"
BSSPACE = re.compile(r'\\ (?:geq|leq|cdot|frac|sqrt|times|circ|equiv|mathbf|neq|dots|pi)\b')
ENVBR = re.compile(r'\\(?:begin|end)\{ |\\(?:begin|end)\{[^}]*[' + VNL + r'][^}]*\}')
TIME = re.compile(r'\\time(?![a-zA-Z])')
TWOX = re.compile(r'(?<![\w.])2x(?!\w)')
DECCOMMA_MATH = re.compile(r'\$[^$]*\d,\d[^$]*\$')
ASYTR = re.compile(r'nhãn\(|dấu chấm\(|thư mục\(')
mode = sys.argv[2]
if mode == 'test':
    rows = [x['row'] for x in json.load(open(sys.argv[1] + '/work/samples_mm.json', encoding='utf-8'))]
else:
    rows = (json.loads(l) for l in open('E:/MATH-REASONING-THESIS/vi-math-smt/data/test/metaMATH/metaMathQA-395K_json_vi.jsonl', encoding='utf-8'))
C = collections.Counter(); ex = collections.defaultdict(list)
def hit(name, cond, i):
    if cond:
        C[name] += 1; ex[name].append(i)
for i, r in enumerate(rows):
    qe, qv, re_, rv = r['query_en'], r['query_vi'], r['response_en'], r['response_vi']
    C['rows'] += 1
    hit('q_macro_translated', MACRO.search(qv) or MACRO2.search(qv), i)
    hit('r_macro_translated', MACRO.search(rv) or MACRO2.search(rv), i)
    hit('any_macro_translated', MACRO.search(qv) or MACRO2.search(qv) or MACRO.search(rv) or MACRO2.search(rv), i)
    hit('q_time_macro', TIME.search(qv) and not TIME.search(qe), i)
    hit('any_bs_space_macro', BSSPACE.search(qv + rv) and not BSSPACE.search(qe + re_), i)
    hit('any_env_broken', ENVBR.search(qv + rv) and not ENVBR.search(qe + re_), i)
    if '[asy]' in qe:
        C['asy_en'] += 1
        hit('asy_translated', ASYTR.search(qv), i)
    if '\\boxed' in re_:
        C['boxed_en'] += 1
        hit('boxed_lost', '\\boxed' not in rv, i)
    hit('decimal_comma_in_math', DECCOMMA_MATH.search(rv) and not DECCOMMA_MATH.search(re_), i)
    hit('2x_to_gapdoi', TWOX.search(re_) and 'gấp đôi' in rv and not re.search(r'twice|double', re_, re.I), i)
    hit('dollar_count_changed_q', qe.count('$') != qv.count('$'), i)
print(json.dumps(dict(C), ensure_ascii=False, indent=0))
if mode == 'test':
    print({k: v for k, v in ex.items()})
else:
    rng = random.Random(20261107)
    json.dump({k: rng.sample(v, min(30, len(v))) for k, v in ex.items()}, open(sys.argv[1] + '/work/mm_scan3_ex.json', 'w'))
