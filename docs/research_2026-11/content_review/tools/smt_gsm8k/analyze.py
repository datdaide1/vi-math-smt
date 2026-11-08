"""Full-corpus feature extraction for gsm8k-smt-nvidia-FULL.jsonl (read-only on data).

Usage: python analyze.py <repo_root> <out_jsonl>
"""
import json, re, sys, time
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
import z3  # noqa
from vi_math_verified_aug.verify.uniqueness import classify_verbose, _balanced, _DECL_RE  # noqa
from vi_math_verified_aug.verify.z3_utils import strip_commands  # noqa

TIMEOUT = 10000
NUM = r"-?\d+(?:\.\d+)?"
ANS_LIT = re.compile(rf"^\(assert \(= (?:answer ({NUM})|({NUM}) answer)\)\)$")
TOK = re.compile(r"[^\s()]+|[()]")


def zval(v):
    if v is None:
        return None
    if z3.is_rational_value(v):
        return Fraction(v.numerator_as_long(), v.denominator_as_long())
    if z3.is_int_value(v):
        return Fraction(v.as_long())
    if z3.is_algebraic_value(v):
        return Fraction(str(v.approx(12)).rstrip("?"))
    return None


def solve(smt):
    s = z3.Solver(); s.set("timeout", TIMEOUT)
    try:
        s.add(z3.parse_smt2_string(strip_commands(smt)))
    except Exception as e:  # noqa
        return "parse_error", None, None, str(e)[:120]
    r = s.check()
    if r != z3.sat:
        return str(r), None, s, None
    m = s.model()
    d = next((d for d in m.decls() if d.name() == "answer"), None)
    if d is None:
        return "no_var", None, s, None
    return "sat", zval(m[d]), s, None


def unique_after_drop(smt, decls, asserts, val):
    """Drop asserts that pin answer to a literal; is answer still unique?"""
    keep = [a for a in asserts if not ANS_LIT.match(a)]
    dropped = len(asserts) - len(keep)
    header = "\n".join(f"(declare-fun {n} () {srt})" for n, srt in decls.items())
    s = z3.Solver(); s.set("timeout", TIMEOUT)
    try:
        s.add(z3.parse_smt2_string(header + "\n" + "\n".join(keep)))
    except Exception:
        return dropped, None
    if s.check() != z3.sat:
        return dropped, None
    if "answer" not in decls:
        return dropped, None
    m = s.model()
    a = next((d for d in m.decls() if d.name() == "answer"), None)
    if a is None:
        return dropped, False  # answer unconstrained
    av = m[a]
    s.add(a() != av)
    r = s.check()
    if r == z3.unsat:
        return dropped, True
    if r == z3.sat:
        return dropped, False
    return dropped, None


def parse_sexpr(s):
    toks = TOK.findall(s)
    pos = 0
    def rec():
        nonlocal pos
        t = toks[pos]; pos += 1
        if t == "(":
            out = []
            while toks[pos] != ")":
                out.append(rec())
            pos += 1
            return out
        return t
    return rec()


def syms(e):
    if isinstance(e, list):
        out = set()
        for x in e[1:] if e and isinstance(e[0], str) else e:
            out |= syms(x)
        return out
    return {e} if re.match(r"[A-Za-z_]", e) else set()


def lits(e):
    if isinstance(e, list):
        out = []
        for x in e:
            out += lits(x)
        return out
    return [e] if re.match(NUM + "$", e) else []


def straight_line(asserts, decls):
    """True if every assert is (= v expr) defining a fresh var from already-defined ones
    (an assignment program); returns (is_straight, n_inequalities, n_implicit)."""
    defs = {}
    ineq = 0
    others = []
    for a in asserts:
        e = parse_sexpr(a)[1]
        if isinstance(e, list) and e[0] in ("<", "<=", ">", ">=", "distinct", "not", "and", "or"):
            ineq += 1; continue
        if isinstance(e, list) and e[0] == "=" and len(e) == 3:
            l, r = e[1], e[2]
            if isinstance(l, str) and l in decls and l not in defs:
                defs[l] = syms(r); continue
            if isinstance(r, str) and r in decls and r not in defs:
                defs[r] = syms(l); continue
        others.append(a)
    # check acyclic ordering
    done, changed = set(), True
    while changed:
        changed = False
        for v, deps in defs.items():
            if v not in done and deps <= done:
                done.add(v); changed = True
    acyclic = len(done) == len(defs)
    return (acyclic and not others and ineq == 0), ineq, len(others)


WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
           "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
           "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "dozen": 12, "half": 0.5,
           "twice": 2, "double": 2, "triple": 3, "thrice": 3, "quarter": 0.25, "third": 3, "fourth": 4,
           "fifth": 5, "sixth": 6, "tenth": 10, "pair": 2, "score": 20}
COMMON = {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 24, 30, 31, 28, 52, 60, 100, 365, 1000, 0.5, 0.25, 0.75,
          0.1, 0.2, 0.01, 3600, 1440, 16, 36, 48, 360, 1.5, 9, 11, 26, 104}


def qnums(q):
    out = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", q):
        try:
            out.add(Fraction(m.group(0).replace(",", "").rstrip(".")))
        except Exception:
            pass
    for m in re.finditer(r"(\d+)\s*/\s*(\d+)", q):
        a, b = int(m.group(1)), int(m.group(2))
        if b:
            out.add(Fraction(a, b))
    for w, v in WORDNUM.items():
        if re.search(rf"\b{w}", q, re.I):
            out.add(Fraction(v).limit_denominator(1000) if isinstance(v, float) else Fraction(v))
    ext = set(out)
    for x in out:
        ext |= {x / 100, x * 100, 1 / x if x else 0, 1 + x / 100, 1 - x / 100}
    return out, ext


def gold_frac(g):
    try:
        return Fraction(str(g).replace(",", "").strip())
    except Exception:
        return None


def work(args):
    r, gold, q = args
    smt = r.get("smt_lib") or ""
    t0 = time.time()
    out = {"index": r["index"], "retries": r.get("retries"), "gold": gold, "n_assert": smt.count("(assert")}
    if "(assert" not in smt:
        out.update(status="not_smt", note=smt[:80]); return out
    status, val, _, err = solve(smt)
    out["status"] = status; out["err"] = err
    g = gold_frac(gold)
    out["value"] = None if val is None else float(val)
    out["match"] = (val is not None and g is not None and abs(float(val) - float(g)) < 1e-4)
    decls = {(m.group(1) or m.group(2)): m.group(3) for m in _DECL_RE.finditer(smt)}
    out["sorts"] = sorted(set(decls.values()))
    out["n_vars"] = len(decls)
    asserts = _balanced(smt, "assert")
    try:
        cls, why = classify_verbose(smt)
    except Exception as e:
        cls, why = "unknown", "crash"
    out["cls"], out["why"] = cls, why
    if status == "sat":
        try:
            dropped, uq = unique_after_drop(smt, decls, asserts, val)
        except Exception:
            dropped, uq = None, None
        out["ans_lit_asserts"], out["unique_wo_anslit"] = dropped, uq
        try:
            sl, ineq, impl = straight_line(asserts, decls)
        except Exception:
            sl, ineq, impl = None, None, None
        out["straight_line"], out["n_ineq"], out["n_implicit"] = sl, ineq, impl
    # literal analysis
    try:
        L = []
        for a in asserts:
            L += lits(parse_sexpr(a))
        Lf = [Fraction(x) for x in L]
    except Exception:
        Lf = []
    qn, qext = qnums(q)
    unexplained = sorted({float(x) for x in Lf if x not in qext and x not in {Fraction(c).limit_denominator(1000) for c in COMMON}})
    out["unexplained_lits"] = unexplained[:10]
    out["gold_lit_in_smt"] = (g is not None and g in set(Lf))
    out["gold_in_question"] = (g is not None and g in qn)
    out["qnums_missing"] = sorted(float(x) for x in qn if x not in set(Lf) and x >= 1 and x not in {Fraction(v) for v in (1, 2, 3, 4)})[:10]
    # trivial: one assert, answer = closed form of literals
    out["closed_form"] = (len(asserts) == 1 and len(decls) == 1)
    out["secs"] = round(time.time() - t0, 3)
    return out


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/formalize_output/gsm8k/gsm8k-smt-nvidia-FULL.jsonl", encoding="utf-8")]
    gold = {}
    for l in open(ROOT / "data/generate/gsm8k/train.jsonl", encoding="utf-8"):
        x = json.loads(l); gold[x["index"]] = x
    jobs = [(r, gold[r["index"]]["ground_truth"], gold[r["index"]]["question"]) for r in rows]
    t = time.time()
    with Pool(8) as p, open(sys.argv[2], "w", encoding="utf-8") as f:
        for o in p.imap_unordered(work, jobs, chunksize=20):
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("done", len(jobs), round(time.time() - t, 1), "s")


if __name__ == "__main__":
    main()
