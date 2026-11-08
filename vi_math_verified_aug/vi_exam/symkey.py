"""SymPy-checked answer keys for Core candidates (key #2 of the 3-key protocol, T5 §3.1 stage 2).

Scope (deliberately narrow, every key is checked by substitution):
  * ``solve_equation``   – "Giải phương trình …" with one unknown (a parameter fixed by "khi m = …" is
                           substituted); real roots, each verified in the original equation (radical domains).
  * ``solve_system``     – "Giải hệ phương trình …" (``\\left\\{ … \\\\ … \\right.`` / ``cases``), 2–3 unknowns.
  * ``eval_expression``  – "Tính giá trị của biểu thức A khi x = 9" (A defined in the stem or inline), or a
                           closed numeric expression "Tính / Rút gọn A = √12 − …" with no free variables.
Anything else -> no key (``None``), reported as coverage.

The LaTeX converter is a small school-level translator (``\\frac``, ``\\sqrt[n]``, powers, braces, implicit
products, Vietnamese decimal comma). It is intentionally strict: an unknown command -> parse failure, never a
guess.
"""
from __future__ import annotations

import re
import threading

import sympy as sp
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application, parse_expr,
                                        standard_transformations)

from .answers import math_spans

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
LETTERS = {c: sp.Symbol(c, real=True) for c in "abcdefghjklmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ"}
def _school_root(a, n):
    """School convention: odd roots of negatives are real (∛-8 = -2), unlike SymPy's principal root."""
    n = sp.sympify(n)
    return sp.real_root(a, n) if n.is_integer and n.is_odd else sp.root(a, n)


LOCAL = dict(LETTERS, sqrt=sp.sqrt, root=_school_root, pi=sp.pi, Abs=sp.Abs)


class ParseError(ValueError):
    pass


def _group(s: str, i: int) -> tuple[str, int]:
    """s[i] == '{' -> (content, index after matching '}')."""
    depth = 0
    for k in range(i, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:k], k + 1
    raise ParseError("unbalanced braces")


def _arg(s: str, i: int) -> tuple[str, int]:
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        raise ParseError("missing argument")
    if s[i] == "{":
        return _group(s, i)
    if s[i] == "\\":
        m = re.match(r"\\[a-zA-Z]+", s[i:])
        return m.group(0), i + len(m.group(0))
    m = re.match(r"\d|[A-Za-z]", s[i:])  # TeX: an undelimited argument is ONE token (\frac12 = 1/2)
    if not m:
        raise ParseError(f"bad argument at {s[i:i + 10]!r}")
    return m.group(0), i + len(m.group(0))


def _conv(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([a-zA-Z]+)", s[i:])
            if not m:
                nxt = s[i + 1:i + 2]
                if nxt in "{}":
                    out.append("(" if nxt == "{" else ")")
                i += 2
                continue
            cmd = m.group(1)
            i += len(m.group(0))
            if cmd in ("frac", "dfrac", "tfrac"):
                a, i = _arg(s, i)
                b, i = _arg(s, i)
                out.append(f"(({_conv(a)})/({_conv(b)}))")
            elif cmd == "sqrt":
                n = None
                if i < len(s) and s[i] == "[":
                    j = s.index("]", i)
                    n, i = s[i + 1:j], j + 1
                a, i = _arg(s, i)
                out.append(f"root(({_conv(a)}),({_conv(n)}))" if n else f"sqrt({_conv(a)})")
            elif cmd in ("cdot", "times"):
                out.append("*")
            elif cmd == "div":
                out.append("/")
            elif cmd == "pi":
                out.append(" pi ")
            elif cmd in ("left", "right", "displaystyle", "limits", "big", "Big", "bigg", "Bigg"):
                if i < len(s) and s[i] in ".":
                    i += 1
            elif cmd in ("rm", "text", "mathrm", "mbox"):
                if i < len(s) and s[i] == "{":
                    _, i = _group(s, i)
            elif cmd in ("quad", "qquad"):
                out.append(" ")
            else:
                raise ParseError(f"unsupported \\{cmd}")
        elif c == "{":
            inner, i = _group(s, i)
            out.append("(" + _conv(inner) + ")")
        elif c == "^":
            out.append("**")
            i += 1
        elif c in "[]":
            out.append("(" if c == "[" else ")")
            i += 1
        elif c == ":":
            out.append("/")
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def latex_to_sympy(tex: str) -> sp.Expr:
    t = tex.strip().rstrip(".;, ")
    t = re.sub(r"\\[,;!: ]|\\quad|\\qquad|~|\\mkern\s*\d+mu", " ", t)
    t = re.sub(r"([a-zA-Z])_\{?\s*(\d)\s*\}?", r"\1_\2", t)   # x_{1} -> x_1 (one symbol)
    # Vietnamese convention: the decimal separator is the comma; '.' between digits is a product (4.5 = 4·5)
    t = re.sub(r"\{\s*,\s*\}", "§", t)
    t = re.sub(r"(?<=\d),(?=\d)", "§", t)
    t = re.sub(r"\^\s*\\circ|\^\s*\{\s*\\circ\s*\}|°", "", t)
    t = t.replace("\\left|", "Abs(").replace("\\right|", ")")
    try:
        py = _conv(t)
    except ParseError:
        raise
    except Exception as e:  # noqa: BLE001  (malformed LaTeX -> parse failure, never a crash)
        raise ParseError(f"conv: {type(e).__name__}") from e
    if re.search(r"[=<>≤≥]", py):
        raise ParseError("relation inside expression")
    py = re.sub(r"(?<=[\d\)])\s*\.\s*(?=[\d\(a-zA-Z])", "*", py).replace("§", ".")
    try:
        out = parse_expr(py, local_dict=dict(LOCAL), transformations=TRANSFORMS, evaluate=True)
    except Exception as e:  # noqa: BLE001
        raise ParseError(f"parse_expr: {e}") from e
    if not isinstance(out, sp.Expr):
        raise ParseError(f"not an expression: {type(out).__name__}")
    return out


def parse_relation(tex: str) -> tuple[sp.Expr, sp.Expr]:
    t = re.sub(r"\\left|\\right", "", tex).strip().rstrip(".;, ")
    if t.count("=") != 1 or re.search(r"\\ne|\\le|\\ge|<|>", t):
        raise ParseError("not a single equation")
    a, b = t.split("=")
    return latex_to_sympy(a), latex_to_sympy(b)


class Timeout(Exception):
    pass


def run_with_timeout(fn, seconds: float = 8.0):
    res: dict = {}

    def target():
        try:
            res["v"] = fn()
        except Exception as e:  # noqa: BLE001
            res["e"] = e

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        raise Timeout()
    if "e" in res:
        raise res["e"]
    return res["v"]


# --------------------------------------------------------------------------- key builders
RE_SYS = re.compile(r"\\begin\{(?:array|cases|aligned)\}(?:\{[^}]*\})*(.*?)\\end\{(?:array|cases|aligned)\}", re.S)
RE_ASSIGN = re.compile(r"^\s*([a-zA-Z])\s*=\s*(.+)$")


def _assignments(text: str) -> dict:
    """'khi m = 2', 'tại x = 9', 'với x = 16' -> {m: 2}."""
    out = {}
    for span in math_spans(text):
        for part in re.split(r"[;]|\\text\{ và \}", span):
            m = RE_ASSIGN.match(re.sub(r"\\left|\\right", "", part).strip().rstrip(".,"))
            if m:
                try:
                    v = latex_to_sympy(m.group(2))
                    if not v.free_symbols:
                        out[LETTERS.get(m.group(1), sp.Symbol(m.group(1), real=True))] = v
                except ParseError:
                    pass
    return out


def _verify_root(lhs, rhs, var, val) -> bool:
    try:
        d = (lhs - rhs).subs(var, val)
        dn = complex(sp.N(d, 30))
        return abs(dn) < 1e-9
    except Exception:  # noqa: BLE001
        return False


def solve_equation(ask: str, stem: str) -> dict | None:
    eqs = [s for s in math_spans(ask) if s.count("=") == 1 and "begin" not in s]
    use_stem = False
    if not eqs and re.search(r"phương trình\s*(\\\(\s*)?\(\s*\d\s*\)", ask):
        eqs = [s for s in math_spans(stem) if s.count("=") == 1 and "begin" not in s]
        use_stem = True
    subs = _assignments(ask)
    for e in eqs:
        try:
            lhs, rhs = parse_relation(e.replace("(1)", "").replace("\\left( 1 \\right)", ""))
        except ParseError:
            continue
        lhs, rhs = lhs.subs(subs), rhs.subs(subs)
        free = sorted((lhs - rhs).free_symbols, key=str)
        if len(free) != 1:
            continue
        x = free[0]
        try:
            sols = run_with_timeout(lambda: sp.solveset(sp.Eq(lhs, rhs), x, domain=sp.S.Reals))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(sols, sp.FiniteSet):
            continue
        roots = [r for r in sols if _verify_root(lhs, rhs, x, r)]
        return {"key_type": "solve_equation", "variable": str(x), "equation": e, "from_stem": use_stem,
                "substituted": {str(k): str(v) for k, v in subs.items()},
                "values": sorted({sp.nsimplify(r) for r in roots}, key=lambda v: float(sp.N(v))),
                "n_roots": len(roots)}
    return None


def solve_system(ask: str, stem: str) -> dict | None:
    for txt in (ask, stem):
        for span in math_spans(txt):
            m = RE_SYS.search(span)
            if not m:
                continue
            rows = [r for r in re.split(r"\\\\", m.group(1)) if r.strip()]
            rels = []
            try:
                for r in rows:
                    r = re.sub(r"&", "", r).strip().strip("{}").strip()
                    rels.append(parse_relation(r))
            except ParseError:
                continue
            subs = _assignments(ask)
            eqs = [sp.Eq(a.subs(subs), b.subs(subs)) for a, b in rels]
            free = sorted(set().union(*[e.free_symbols for e in eqs]), key=str)
            if not (2 <= len(free) <= 3) or len(free) > len(eqs):
                continue
            try:
                sol = run_with_timeout(lambda: sp.solve(eqs, free, dict=True))
            except Exception:  # noqa: BLE001
                continue
            good = []
            for d in sol:
                if all(v.is_real is not False and not v.free_symbols for v in d.values()):
                    try:
                        ok = all(abs(complex(sp.N((e.lhs - e.rhs).subs(d), 30))) < 1e-9 for e in eqs)
                    except Exception:  # noqa: BLE001
                        ok = False
                    if ok and all(v in d for v in free):
                        good.append(tuple(sp.nsimplify(d[v]) for v in free))
            return {"key_type": "solve_system", "variables": [str(v) for v in free], "system": span,
                    "values": sorted(set(good), key=lambda t: [float(sp.N(v)) for v in t]),
                    "n_solutions": len(good)}
    return None


RE_DEF = re.compile(r"^\s*([A-Z])\s*(?:\(x\)|\\left\(\s*x\s*\\right\))?\s*=\s*(.+)$", re.S)


RE_INVERSE = re.compile(r"(tìm|tính|xác định)\s+(tất cả\s+)?(các\s+)?(giá trị\s+)?(nguyên\s+)?(của\s+)?\\?\(?\s*[a-z]"
                        r"\s*\\?\)?\s*(để|sao cho|thỏa|thoả)", re.I)


def strip_condition(rhs: str) -> str:
    """'… \\,\\,( x \\ge 0, x \\ne 25 )' / '…, x > 0' / '… với x ≥ 0' appended to a definition -> removed."""
    rhs = re.sub(r"(\\,|\\;|\\quad|\s)*\(\s*\{?\s*[a-z]\s*(\\geq?|\\neq?|\\leq?|>|<|\\geqslant|\\leqslant)[^()]*\}?\s*\)"
                 r"\s*[.,;]?\s*$", "", rhs)
    rhs = re.sub(r"[,;](\\,|\\;|\s)*(với\s*)?[a-z]\s*(\\geq?|\\neq?|\\leq?|>|<|\\geqslant|\\leqslant).*$", "", rhs)
    return rhs.strip()


def eval_expression(ask: str, stem: str) -> dict | None:
    if not re.search(r"(tính|giá trị|rút gọn|thu gọn)", ask, re.I) or RE_INVERSE.search(ask):
        return None
    if re.search(r"x_1|x_\{1\}", ask):
        return None
    defs = {}
    for span in math_spans(stem) + math_spans(ask):
        m = RE_DEF.match(re.sub(r"\\left|\\right", "", span).strip())
        if m:
            rhs = strip_condition(m.group(2))
            if "=" in rhs:
                continue
            try:
                defs[m.group(1)] = (span, latex_to_sympy(rhs))
            except ParseError:
                pass
    if not defs:
        return None
    named = (re.findall(r"biểu thức\s*:?\s*\\\(\s*([A-Z])\s*[.,;:]?\s*\\\)", ask)
             or re.findall(r"\\\(\s*([A-Z])\s*[.,;:]?\s*\\\)", ask)
             or re.findall(r"biểu thức\s+([A-Z])\b", ask))
    if named and not any(n in defs for n in named):
        return None  # the asked-for expression did not parse: never fall back to another one
    name = next((n for n in named if n in defs), None)
    if name is None:
        inline = [n for n in defs if re.search(rf"\\\(\s*{n}\s*=", ask)]
        name = inline[0] if inline else (list(defs)[0] if len(defs) == 1 else None)
    if name is None:
        return None
    span, expr = defs[name]
    subs = _assignments(re.sub(rf"\\\(\s*{name}\s*=.*?\\\)", "", ask))
    val = expr.subs(subs)
    if val.free_symbols:
        if re.search(r"rút gọn|thu gọn", ask, re.I) and len(val.free_symbols) == 1:
            # simplification: the key is the defining expression; answers are checked for equivalence
            return {"key_type": "simplify_equivalence", "name": name, "definition": span,
                    "variable": str(next(iter(val.free_symbols))), "values": [val],
                    "domain_text": stem + " " + ask}
        return None
    try:
        v = run_with_timeout(lambda: sp.nsimplify(sp.simplify(val)), 8)
    except Exception:  # noqa: BLE001
        return None
    if v.is_real is False or v.has(sp.zoo, sp.nan):
        return None
    return {"key_type": "eval_expression", "name": name, "definition": span,
            "substituted": {str(k): str(x) for k, x in subs.items()}, "values": [v]}


def _domain_ok(text: str, var: sp.Symbol, val) -> bool:
    """Respect stated conditions like 'x \\ge 0, x \\ne 25' / 'x > 0' found in the stem or the ask."""
    ops = {"\\ne": lambda v: v != 0, "\\neq": lambda v: v != 0, "\\ge": lambda v: v >= 0,
           "\\geq": lambda v: v >= 0, "\\geqslant": lambda v: v >= 0, "\\le": lambda v: v <= 0,
           "\\leq": lambda v: v <= 0, "\\leqslant": lambda v: v <= 0, ">": lambda v: v > 0, "<": lambda v: v < 0}
    for span in math_spans(text):
        # compound 'a \le x \le b' / 'a < x < b'
        mc = re.match(rf"^\s*(.+?)\s*(\\leq|\\le|\\leqslant|<)\s*{re.escape(str(var))}\s*(\\leq|\\le|\\leqslant|<)\s*(.+?)\s*\.?$",
                      re.sub(r"\\left|\\right", "", span))
        if mc:
            try:
                lo, hi = latex_to_sympy(mc.group(1)), latex_to_sympy(mc.group(4))
                if not lo.free_symbols and not hi.free_symbols:
                    v = float(sp.N(val))
                    if not (ops[mc.group(2)](float(sp.N(lo)) - v) if mc.group(2) != "<" else v > float(sp.N(lo))):
                        return False
                    if not (ops[mc.group(3)](v - float(sp.N(hi))) if mc.group(3) != "<" else v < float(sp.N(hi))):
                        return False
                    continue
            except (ParseError, TypeError, ValueError):
                pass
        for part in re.split(r"[;,]|\\text\{\s*và\s*\}", re.sub(r"\\left|\\right", "", span)):
            m = re.match(rf"^\s*{re.escape(str(var))}\s*(\\neq|\\ne|\\geqslant|\\geq|\\ge|\\leqslant|\\leq|\\le|>|<)"
                         rf"\s*(.+)$", part.strip().rstrip("."))
            if not m:
                continue
            try:
                b = latex_to_sympy(m.group(2))
            except ParseError:
                continue
            if b.free_symbols:
                continue
            if not ops[m.group(1)](float(sp.N(val - b))):
                return False
    return True


def solve_for_value(ask: str, stem: str) -> dict | None:
    """'Tìm x để P = 1/6' (P defined in the stem): real roots, checked by substitution and stated domain."""
    if not RE_INVERSE.search(ask) or re.search(r"nguyên|lớn nhất|nhỏ nhất|<|>|\\le|\\ge|\\ne", ask):
        return None
    defs = {}
    for span in math_spans(stem):
        m = RE_DEF.match(re.sub(r"\\left|\\right", "", span).strip())
        if m and "=" not in strip_condition(m.group(2)):
            try:
                defs[m.group(1)] = latex_to_sympy(strip_condition(m.group(2)))
            except ParseError:
                pass
    for span in math_spans(ask):
        m = re.match(r"^\s*([A-Z])\s*=\s*(.+)$", re.sub(r"\\left|\\right", "", span).strip())
        if not m or m.group(1) not in defs:
            continue
        try:
            target = latex_to_sympy(m.group(2))
        except ParseError:
            continue
        expr = defs[m.group(1)]
        free = sorted(expr.free_symbols | target.free_symbols, key=str)
        if len(free) != 1:
            continue
        x = free[0]
        try:
            sols = run_with_timeout(lambda: sp.solveset(sp.Eq(expr, target), x, domain=sp.S.Reals))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(sols, sp.FiniteSet):
            continue
        roots = [r for r in sols if _verify_root(expr, target, x, r) and _domain_ok(stem + " " + ask, x, r)]
        return {"key_type": "solve_for_value", "variable": str(x), "name": m.group(1), "target": span,
                "values": sorted({sp.nsimplify(r) for r in roots}, key=lambda v: float(sp.N(v)))}
    return None


def vieta_eval(ask: str, stem: str) -> dict | None:
    """'Gọi x1, x2 là hai nghiệm của x^2+7x+3=0 … tính M = x1^2 + x2^2' -> evaluate on the actual roots.
    Defined only for a symmetric target (same value under x1<->x2) and a parameter-free (or 'khi m = …') equation."""
    text = stem + "\n" + ask
    if not re.search(r"x_1|x_\{1\}", text) or not re.search(r"tính|giá trị", ask, re.I):
        return None
    eq = eq_span = None
    for span in math_spans(text):
        flat = re.sub(r"[{}\s]", "", span)
        if span.count("=") == 1 and "x_" not in flat and re.search(r"x\^", flat):
            try:
                eq, eq_span = parse_relation(span), span
                break
            except ParseError:
                continue
    if eq is None:
        return None
    x = LETTERS["x"]
    poly = (eq[0] - eq[1]).subs(_assignments(ask))
    if poly.free_symbols != {x}:
        return None
    target = None
    for span in reversed(math_spans(ask)):
        m = re.match(r"^\s*([A-Z])\s*=\s*(.+)$", re.sub(r"\\left|\\right", "", span).strip(), re.S)
        if m and "x_" in re.sub(r"[{}\s]", "", m.group(2)):
            try:
                target = (m.group(1), latex_to_sympy(m.group(2)))
                break
            except ParseError:
                continue
    if target is None:
        return None
    try:
        P = sp.Poly(poly, x)
        if P.degree() != 2:
            return None
        roots = sp.solve(poly, x)
    except Exception:  # noqa: BLE001
        return None
    if len(roots) != 2 or any(not r.is_real for r in roots):
        return None
    e = target[1]
    syms = {str(s): s for s in e.free_symbols}
    if set(syms) - {"x_1", "x_2"}:
        return None
    x1, x2 = syms.get("x_1", sp.Symbol("x_1")), syms.get("x_2", sp.Symbol("x_2"))
    v1 = sp.nsimplify(sp.simplify(e.subs({x1: roots[0], x2: roots[1]})))
    v2 = sp.nsimplify(sp.simplify(e.subs({x1: roots[1], x2: roots[0]})))
    if sp.simplify(v1 - v2) != 0:
        return None  # order-dependent target: needs the page's x1/x2 convention -> human/Claude key
    return {"key_type": "vieta_eval", "equation": eq_span, "name": target[0], "values": [v1]}


def build_key(ask: str, stem: str, ans_type: str, tps: list[str]) -> dict | None:
    if ans_type not in ("solution_set", "numeric", "expression", "parameter_condition"):
        return None
    if ans_type == "solution_set":
        fns = (solve_system, solve_equation) if re.search(r"hệ", ask, re.I) else (solve_equation, solve_system)
    else:
        fns = (vieta_eval, solve_for_value, eval_expression)
    for fn in fns:
        try:
            k = fn(ask, stem)
        except Exception:  # noqa: BLE001
            k = None
        if k:
            k["values_str"] = [str(v) for v in k["values"]]
            if any("RootOf" in v for v in k["values_str"]):
                return None  # no closed form: almost always a mis-parsed definition, never a school answer
            k["values"] = [[str(x) for x in v] if isinstance(v, tuple) else str(v) for v in k["values"]]
            return k
    return None


# --------------------------------------------------------------------------- comparison
RE_LHS = re.compile(r"(?:(?<=^)|(?<=[\s;,{]))(\(\s*\{?[a-zA-Z_0-9\s;,]+\}?\s*\)|[a-zA-Z](?:_\d)?)\s*=\s*")


def _clean_answer(ans: str) -> str:
    s = re.sub(r"\\left|\\right|\\mkern\s*\d+mu|\\displaystyle", "", ans)
    s = re.sub(r"\\text\{[^}]*\}|\\rm\{[^}]*\}|\\,|\\;|\\!|\\quad|\\ |\\tfrac", lambda m: "\\frac" if m.group(0) ==
               "\\tfrac" else " ", s)
    # '\left[ \begin{array}{l} x_1 = 1 \\ x_2 = 2 \end{array} \right.' -> 'x_1 = 1; x_2 = 2'
    s = re.sub(r"\\begin\{(?:array|cases|aligned)\}(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})?|\\end\{(?:array|cases|aligned)\}",
               " ", s)
    s = s.replace("\\\\", ";")
    s = s.replace("\\{", "{").replace("\\}", "}")
    s = re.sub(r"^\s*\[", "", s)
    s = re.sub(r"\{\s*\}", "", s)
    s = re.sub(r"([a-zA-Z])_\{\s*(\d)\s*\}", r"\1_\2", s)
    for _ in range(3):  # '{x_1}' -> 'x_1', but never '\sqrt{x}' -> '\sqrtx'
        s = re.sub(r"(?<![a-zA-Z])\{\s*([a-zA-Z](?:_\d)?)\s*\}", r"\1", s)
    return s.strip().rstrip(".;, ")


def _trim_unbalanced(s: str) -> str:
    s = s.strip().rstrip(".;, {")
    while s and s[-1] in "})" and s.count("{") + s.count("(") < s.count("}") + s.count(")"):
        s = s[:-1].rstrip(".;, {")
    while s and s[0] in "{(" and s.count("{") + s.count("(") > s.count("}") + s.count(")"):
        s = s[1:].lstrip()
    return s


def _split_items(rhs: str, comma_sep: bool = False) -> list[str]:
    rhs = _trim_unbalanced(rhs)
    while len(rhs) >= 2 and rhs[0] in "{(" and rhs[-1] in "})" and _balanced(rhs[1:-1]):
        rhs = rhs[1:-1].strip()
    out, depth, cur = [], 0, []
    for k, ch in enumerate(rhs):
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        decimal = k > 0 and rhs[k - 1].isdigit() and k + 1 < len(rhs) and rhs[k + 1].isdigit() and not comma_sep
        if depth == 0 and (ch == ";" or (ch == "," and not decimal)):
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [o.strip() for o in out if o.strip()]


def _balanced(s: str) -> bool:
    d = 0
    for ch in s:
        d += ch in "{("
        d -= ch in "})"
        if d < 0:
            return False
    return d == 0


def answer_values(ans: str) -> list[sp.Expr] | None:
    """'x = 2; x = 3' / 'S = {2;3}' / '(x;y) = (2;1)' / 'x_1 = 6; x_2 = 5' / 'A = 1' -> numeric values."""
    if not ans:
        return None
    s = _clean_answer(ans)
    marks = list(RE_LHS.finditer(s))
    rhss = []
    if marks:
        for k, m in enumerate(marks):
            end = marks[k + 1].start() if k + 1 < len(marks) else len(s)
            # '(x, y) = (-1, 1)': a comma-separated tuple on the left means commas separate on the right too
            rhss.append((s[m.end():end].strip().rstrip(";, ").removesuffix("và").strip(),
                         m.group(1).startswith("(") and "," in m.group(1)))
    else:
        rhss = [(s, False)]
    vals = []
    for rhs, comma_sep in rhss:
        if "=" in rhs:  # chained 'A = … = 3': last member
            rhs = rhs.split("=")[-1]
        for tok in _split_items(rhs, comma_sep):
            tok = re.sub(r"\s*và\s*$", "", tok).strip()
            if not tok or tok in ("\\emptyset", "\\varnothing"):
                continue
            try:
                if "\\pm" in tok:
                    a, b = tok.split("\\pm", 1)
                    A = latex_to_sympy(a) if a.strip() else sp.Integer(0)
                    B = latex_to_sympy(b)
                    vals += [A + B, A - B]
                    continue
                v = latex_to_sympy(tok)
            except ParseError:
                return None
            if v.free_symbols:
                return None
            vals.append(v)
    return vals or None


SAMPLE_POINTS = [sp.Rational(n, d) for n, d in ((23, 10), (37, 10), (71, 10), (119, 10), (163, 10), (41, 100),
                                                 (529, 100), (1009, 10), (3, 7), (13, 10), (17, 10), (19, 10),
                                                 (-7, 10), (-23, 10), (-41, 10))]


def _equivalent(expr: sp.Expr, ans: str, var: str, domain_text: str) -> str:
    s = _clean_answer(ans)
    if "=" in s:
        s = s.split("=")[-1]
    try:
        a = latex_to_sympy(s)
    except ParseError:
        return "unparsed"
    x = LETTERS.get(var, sp.Symbol(var, real=True))
    if a.free_symbols - {x}:
        return "unparsed"
    n_ok = 0
    for p in SAMPLE_POINTS:
        if not _domain_ok(domain_text, x, p):
            continue
        try:
            u, v = complex(sp.N(expr.subs(x, p), 30)), complex(sp.N(a.subs(x, p), 30))
        except Exception:  # noqa: BLE001
            continue
        if abs(u.imag) > 1e-12 or abs(v.imag) > 1e-12 or u != u or v != v:
            continue
        if abs(u - v) > 1e-8 * max(1, abs(u)):
            return "disagree"
        n_ok += 1
    return "agree" if n_ok >= 3 else "unparsed"


def compare(key: dict, ans: str | None) -> str:
    """agree | disagree | unparsed | no_key."""
    if not key:
        return "no_key"
    if not ans:
        return "unparsed"
    ans = prep_answer(ans)
    if key["key_type"] == "simplify_equivalence":
        return _equivalent(sp.sympify(key["values_str"][0], locals=dict(LETTERS)), ans, key["variable"],
                           key.get("domain_text", ""))
    av = answer_values(ans)
    if av is None:
        return "unparsed"
    flat_k = []
    for v in key["values_str"]:
        if v.startswith("("):
            flat_k += [sp.sympify(x) for x in v.strip("()").split(",") if x.strip()]
        else:
            flat_k.append(sp.sympify(v))
    if not flat_k:  # solver found no real solution: answers like 'vô nghiệm' are not parsed as numbers
        return "unparsed"
    try:
        fa = sorted(float(sp.N(x)) for x in av)
        fk = sorted(float(sp.N(x)) for x in flat_k)
    except Exception:  # noqa: BLE001
        return "unparsed"
    if len(fa) != len(fk):
        if sorted(set(round(x, 6) for x in fa)) == sorted(set(round(x, 6) for x in fk)):
            return "agree"
        return "disagree"
    return "agree" if all(abs(a - b) < 1e-6 * max(1, abs(b)) for a, b in zip(fa, fk)) else "disagree"


def prep_answer(x: str) -> str:
    """Normalise a free-form final answer: drop \\( \\) / $ wrappers, 'x \\in {…}' -> 'S = {…}', 'hoặc'/'và' -> ';',
    trailing units, Vietnamese thousands dots outside formulas."""
    x = re.sub(r"\\\(|\\\)|\\\[|\\\]|\$", " ", x)
    x = re.sub(r"\\(?:text|rm|mathrm)\s*\{[^}]*\}\s*(\^\s*\{?\s*\d\s*\}?)?", " ", x)   # '\text{ cm}^2' (units)
    x = re.sub(r"\{\s*\\rm\s*\{[^}]*\}\s*\}(\^\s*\{?\s*\d\s*\}?)?", " ", x)
    for _ in range(3):  # '350\,000' (thin-space thousands grouping)
        x = re.sub(r"(\d)\\,(\d{3})(?!\d)", r"\1\2", x)
    x = re.sub(r"\\[ ,;!]", " ", x)
    x = re.sub(r"(?<![A-Za-z\\])[a-zA-Z]\s*\\in\s*", "S = ", x)
    x = re.sub(r"\bhoặc\b|\bvà\b", ";", x)
    # provisional joins like '(x;y); (2;1)': drop the letter-only tuple
    x = re.sub(r"(\\left)?\(\s*\{?\s*[a-z]\s*([;,]\s*[a-z]\s*)+\}?\s*(\\right)?\)\s*;", "", x)
    x = _strip_units(x.strip())
    x = re.sub(r"(?<=[\d\}\)])\s+[^\d\\{}()=;,]*$", "", x)  # trailing words: '30 bộ' -> '30'
    return _vn_number(x.strip())


def _strip_units(s: str) -> str:
    s = re.sub(r"\\(?:rm|text|mathrm)\{[^}]*\}", " ", s)
    s = re.sub(r"(?<=[\d\}\)])\s*(?:k?m\^?\{?[23]?\}?|cm\^?\{?[23]?\}?|dm|mm|đồng|nghìn đồng|triệu đồng|giờ|phút|"
               r"km/h|m/s|lít|kg|g|%|ngày|người|học sinh|sản phẩm|chiếc|cái|quyển|tuổi|xe|bạn)\b(?=[^0-9]*$).*$",
               "", s)
    return s


def _vn_number(s: str) -> str:
    """'140.000' (Vietnamese thousands separator, outside LaTeX) -> '140000'."""
    return re.sub(r"(?<!\d)(\d{1,3})((?:\.\d{3})+)(?!\d)", lambda m: m.group(1) + m.group(2).replace(".", ""), s)


def _loose_numbers(s: str) -> list[float] | None:
    """Numbers in a worded answer ('Chị 12 tuổi, em 6 tuổi'); None if the text has relations/variables in math."""
    if re.search(r"[<>]|\\le|\\ge|\\ne|\\sqrt|\\frac|\\pi", s):
        return None
    nums = re.findall(r"-?\d+(?:,\d+)?", _vn_number(s))
    return [float(n.replace(",", ".")) for n in nums] or None


def agree_free(a: str | None, b: str | None, domain_text: str = "", numeric_fallback: bool = False) -> str:
    """Compare two free-form final answers (provisional vs Claude): numeric multiset, else single-variable
    expression equivalence by sampling. agree | partial | disagree | unparsed. 'partial' = one answer's numbers
    are a strict subset of the other's (e.g. the provisional extractor kept one of two requested quantities)."""
    if not a or not b:
        return "unparsed"
    a2, b2 = prep_answer(a), prep_answer(b)
    va, vb = answer_values(a2), answer_values(b2)
    fa = fb = None
    if va is not None and vb is not None:
        try:
            fa = sorted(round(float(sp.N(x)), 6) for x in va)
            fb = sorted(round(float(sp.N(x)), 6) for x in vb)
        except Exception:  # noqa: BLE001
            return "unparsed"
    elif numeric_fallback:
        la = [round(float(sp.N(x)), 6) for x in va] if va is not None else _loose_numbers(a2)
        lb = [round(float(sp.N(x)), 6) for x in vb] if vb is not None else _loose_numbers(b2)
        if la is not None and lb is not None:
            fa, fb = sorted(la), sorted(lb)
    if fa is not None and fb is not None:
        sa, sb = set(fa), set(fb)
        if sa == sb:
            return "agree"
        if len(fa) == len(fb) and all(abs(x - y) < 1e-6 * max(1, abs(y)) for x, y in zip(fa, fb)):
            return "agree"
        if sa < sb or sb < sa:
            return "partial"
        return "disagree"
    # expressions in one variable (Rút gọn)
    try:
        ea = latex_to_sympy(_clean_answer(a2).split("=")[-1])
        eb = latex_to_sympy(_clean_answer(b2).split("=")[-1])
    except (ParseError, Exception):  # noqa: BLE001
        return "unparsed"
    fs = ea.free_symbols | eb.free_symbols
    if len(fs) != 1:
        return "unparsed"
    return _equivalent(ea, b2, str(next(iter(fs))), domain_text)
