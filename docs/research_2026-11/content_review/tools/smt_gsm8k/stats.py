import json, random, sys, math, collections as C
SP = sys.argv[1]; sys.path.insert(0, SP)
from ann1 import A
from mutman import OVERRIDE
F = {json.loads(l)['index']: json.loads(l) for l in open(SP + '/features.jsonl', encoding='utf-8')}
M = {json.loads(l)['index']: json.loads(l) for l in open(SP + '/mut.jsonl')}
S = sorted(random.Random(20261107).sample(range(7473), 150))
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return f"{k}/{n} = {100*p:.1f}% [{100*(c-h):.1f}, {100*(c+h):.1f}]"
n = len(S)
codes = C.Counter(c for i in S for c in A[i][0])
print("== per code"); [print(f"  {c:16} {wilson(k, n)}") for c, k in codes.most_common()]
print("  any code       ", wilson(sum(1 for i in S if A[i][0]), n))
# Z3-based checks on sample
print("  Z3 status sat  ", wilson(sum(F[i]['status']=='sat' for i in S), n))
print("  Z3 match gold  ", wilson(sum(bool(F[i].get('match')) for i in S), n))
print("  nonunique after dropping answer literal", wilson(sum(F[i].get('unique_wo_anslit') is False for i in S), n))
sev = C.Counter(A[i][1] for i in S); print("== severity", dict(sev))
worst = lambda i: A[i][1]
usable = sum(1 for i in S if A[i][1] in (None, 'minor')); fix = sum(1 for i in S if A[i][1]=='major'); disc = sum(1 for i in S if A[i][1]=='critical')
print("usable", wilson(usable, n), "fixable", wilson(fix, n), "discard", wilson(disc, n))
faith = sum(A[i][2] for i in S); print("faithful", wilson(faith, n))
der = C.Counter(A[i][3] for i in S); print("deriving", dict(der))
fd_strict = [i for i in S if A[i][2] and A[i][3]=='full' and F[i].get('match') and 'SMT-TRIVIAL' not in A[i][0]]
fd_full = [i for i in S if A[i][2] and A[i][3]=='full' and F[i].get('match')]
fd_len = [i for i in S if A[i][2] and A[i][3] in ('full','partial') and F[i].get('match')]
print("faithful+deriving (full, non-trivial, matches gold)", wilson(len(fd_strict), n))
print("faithful+deriving (full incl. closed-form trivial)", wilson(len(fd_full), n))
print("faithful+deriving (full or partial precomp)", wilson(len(fd_len), n))
impl = [i for i in S if F[i].get('straight_line') is False]
print("non-straight-line (needs actual solving)", wilson(len(impl), n))
impl_good = [i for i in fd_strict if F[i].get('straight_line') is False]
print("faithful-deriving AND implicit", wilson(len(impl_good), n))
print("sorts", C.Counter(tuple(F[i]['sorts']) for i in S))
# mutation levels (manual)
man = {i: OVERRIDE.get(i, (M[i]['mut_level'],))[0] for i in S}
for L in ['L3','L2','L1','L0']: print("mut", L, wilson(sum(man[i]==L for i in S), n))
print("mut L3 & semantic names", wilson(sum(man[i]=='L3' and M[i]['semantic_names'] for i in S), n))
print("mut L3 & semantic & retargetable", wilson(sum(man[i]=='L3' and M[i]['semantic_names'] and M[i]['retargetable'] for i in S), n))
print("mut L3 & faithful-deriving-strict", wilson(sum(man[i]=='L3' and i in fd_strict for i in S), n))
print("heuristic agreement", sum(man[i]==M[i]['mut_level'] for i in S), "/", n)
# classifier calibration vs reading
print("classifier hardcoded vs manual HARDCODE:", sum(F[i]['cls']=='hardcoded' for i in S), sum('SMT-HARDCODE' in A[i][0] for i in S),
      "both", sum(F[i]['cls']=='hardcoded' and 'SMT-HARDCODE' in A[i][0] for i in S))
print("closed_form flag vs manual TRIVIAL", sum(bool(F[i]['closed_form']) for i in S), sum('SMT-TRIVIAL' in A[i][0] for i in S),
      "both", sum(bool(F[i]['closed_form']) and 'SMT-TRIVIAL' in A[i][0] for i in S))
pre_h = {i for i in S if F[i]['unexplained_lits']}; pre_m = {i for i in S if 'X-PRECOMP' in A[i][0]}
print("precomp heuristic vs manual", len(pre_h), len(pre_m), "both", len(pre_h & pre_m))
# proxy for faithful-deriving on full corpus: derived classifier & match & not closed form & no unexplained literal
proxy = lambda x: x.get('cls')=='derived' and x.get('match') and not x.get('closed_form') and not x.get('unexplained_lits')
ps = {i for i in S if proxy(F[i])}
print("proxy on sample", len(ps), "manual fd_strict", len(fd_strict), "both", len(ps & set(fd_strict)))
print("== FULL CORPUS (n=7473)")
N = len(F); allx = list(F.values())
def fw(name, f): print(f"  {name:44} {wilson(sum(1 for x in allx if f(x)), N)}")
fw("SMT-ERROR (parse/not-SMT)", lambda x: x['status']!='sat')
fw("SMT-MISMATCH (sat, value != gold)", lambda x: x['status']=='sat' and not x['match'])
fw("z3_score=1 but not reproducible (err+mism)", lambda x: not x.get('match'))
fw("hardcoded (classifier)", lambda x: x.get('cls')=='hardcoded')
fw("  of which answer-literal-only (1 assert)", lambda x: x.get('why')=='answer_pinned_literal' and x['n_assert']==1)
fw("non-unique after dropping answer literal", lambda x: x.get('unique_wo_anslit') is False)
fw("non-unique excluding hardcodes", lambda x: x.get('unique_wo_anslit') is False and x.get('cls')!='hardcoded')
fw("closed-form single expression (TRIVIAL)", lambda x: x.get('closed_form') and x.get('cls')!='hardcoded')
fw("unexplained literal (precomp proxy)", lambda x: bool(x.get('unexplained_lits')))
fw("straight-line assignment program", lambda x: x.get('straight_line') is True)
fw("non-straight-line (implicit equation/ineq)", lambda x: x.get('straight_line') is False)
fw("Int sort used", lambda x: 'Int' in x.get('sorts', []))
fw("faithful-deriving proxy", proxy)
fw("faithful-deriving proxy & non-straight-line", lambda x: proxy(x) and x.get('straight_line') is False)
for L in ['L3','L2','L1','L0','ERR']:
    print(f"  mut heuristic {L}", wilson(sum(1 for m in M.values() if m['mut_level']==L), N))
print("  mut L3 & semantic & retargetable", wilson(sum(1 for m in M.values() if m['mut_level']=='L3' and m['semantic_names'] and m['retargetable']), N))
R = C.Counter(); H = C.Counter()
for x in allx:
    k = min(x['retries'] or 0, 3); R[k] += 1; H[k] += x.get('cls')=='hardcoded'
print("hardcode by retries", {k: f"{H[k]}/{R[k]}={100*H[k]/R[k]:.1f}%" for k in sorted(R)})
