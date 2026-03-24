"""
SMT Mutation Engine (FIXED + DIVERSE + Z3 SAFE)
"""

import random
from typing import Optional

from smt_parser import SMTParsed, SMTVariable, parse_smt, rebuild_smt
from z3_validator import validate_smt

random.seed(42)


# ============================================================
# UTIL
# ============================================================

def rand_const():
    return round(random.uniform(0.5, 5.0), 2)


def wrap(expr: str):
    k = rand_const()
    return random.choice([
        f"(* (+ {expr} {k}) {k})",
        f"(+ (* {expr} {k}) {k})",
        f"(* {expr} (+ {k} {k}))",
        f"(- (+ {expr} {k}) {k})",
    ])


# ============================================================
# MUTATIONS
# ============================================================

def mutate_structure(parsed: SMTParsed):
    if not parsed.answer_expr:
        return None

    v = SMTVariable(name=f"h_{random.randint(1,99999)}", sort="Real")

    return rebuild_smt(
        parsed,
        extra_vars=[v],
        extra_asserts=[f"(assert (= {v.name} {rand_const()}))"],
        new_answer_expr=f"(+ {parsed.answer_expr} (* {v.name} 2))"
    )


def mutate_expression(parsed: SMTParsed):
    if not parsed.answer_expr:
        return None

    return rebuild_smt(
        parsed,
        new_answer_expr=wrap(parsed.answer_expr)
    )


def mutate_difficulty(parsed: SMTParsed):
    if not parsed.answer_expr:
        return None

    v = SMTVariable(name=f"d_{random.randint(1,99999)}", sort="Real")

    return rebuild_smt(
        parsed,
        extra_vars=[v],
        extra_asserts=[f"(assert (= {v.name} (+ {parsed.answer_expr} 1)))"],
        new_answer_expr=f"(* {parsed.answer_expr} {v.name})"
    )


# ============================================================
# DISPATCHER
# ============================================================

def mutate_single(
    smt_code: str,
    algorithm: str = "auto",
    partner_smt: Optional[str] = None,
):

    parsed = parse_smt(smt_code)

    strategies = [
        mutate_structure,
        mutate_expression,
        mutate_difficulty,
    ]

    random.shuffle(strategies)

    for strat in strategies:
        try:
            mutated = strat(parsed)
            if not mutated:
                continue

            ok, ans, status = validate_smt(mutated)
            if not ok:
                continue

            return {
                "smt_code": mutated,
                "ground_truth": ans,
                "z3_status": status
            }

        except:
            continue

    return None