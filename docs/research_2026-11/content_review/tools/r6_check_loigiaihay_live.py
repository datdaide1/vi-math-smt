import sys, re, json, difflib
from bs4 import BeautifulSoup
sys.stdout.reconfigure(encoding='utf-8')
S = sys.argv[1]
d = json.load(open('E:/MATH-REASONING-THESIS/vi-math-smt/_scratch/vi_exam/loigiaihay_exams.json', encoding='utf-8'))
def norm(t):
    t = re.sub(r'\[\[IMG:[^\]]*\]\]', ' ', t)
    t = re.sub(r'Ad Tag.*$', '', t, flags=re.S)
    return re.sub(r'\s+', ' ', t.replace('\xa0', ' ')).strip()
CHECK = [(0, 'Câu 1', 'Câu 1:', 'Câu 2:'), (0, 'Câu 2', 'Câu 2:', 'Câu 3:'), (2, 'Câu I', 'Câu I (1,5', 'Câu II ('), (2, 'Câu V', 'Câu V (0,5', None),
         (7, 'Bài IV', 'Bài IV (3', 'Bài V ('), (10, 'Câu 5', 'Câu\n 5 (3,0', 'Câu 6 (0,5'), (10, 'Câu 7', 'Câu 7.', 'Câu 8.'),
         (12, 'Câu 3', 'Câu 3.', 'Câu 4.'), (12, 'Câu 13', 'Câu 13.', 'Câu 14.'), (12, 'Câu 14', 'Câu 14.', 'Câu 15.'), (9, 'Câu 1', None, None)]
for ei, cau, a, b in CHECK:
    soup = BeautifulSoup(open(f'{S}/dl_lgh/e{ei}.html', encoding='utf-8').read(), 'html.parser')
    live = soup.select_one('#sub-question-1').get_text('\n')
    it = [x for x in d[ei]['items'] if x['cau'] == cau][0]
    sc = norm(it['problem'])
    if a is None:
        # Lào Cai: show which live 'Câu 1.' blocks exist (PHẦN I vs III)
        idx = [m.start() for m in re.finditer(r'Câu 1\.', live)]
        print(f'e{ei} {cau}: live has {len(idx)} "Câu 1." blocks; scraped text starts:', sc[:90])
        for k in idx: print('    live:', norm(live[k:k+120]))
        continue
    i = live.find(a); j = live.find(b, i + 5) if b else len(live)
    lv = norm(live[i:j if j > 0 else len(live)])
    r = difflib.SequenceMatcher(None, sc, lv, autojunk=False).ratio()
    ops = [(t, sc[i1:i2][:60], lv[j1:j2][:60]) for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sc, lv, autojunk=False).get_opcodes() if t != 'equal']
    print(f'e{ei} {cau}: ratio={r:.3f} len_scraped={len(sc)} len_live={len(lv)} diffs={ops[:4]}')
