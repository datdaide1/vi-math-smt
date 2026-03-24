"""
Z3 Validator - STRICT SMT-LIB VERSION (FIXED)
"""

import z3 as z3_lib
from typing import Tuple, Optional
from config import Z3_TIMEOUT


def validate_smt(smt_code: str) -> Tuple[bool, Optional[str], str]:
    """
    Returns:
        (is_sat, answer, status)
    """

    if not smt_code or not smt_code.strip():
        return False, None, "empty"

    try:
        solver = z3_lib.Solver()
        solver.set("timeout", Z3_TIMEOUT)

        ast = z3_lib.parse_smt2_string(smt_code)
        solver.add(ast)

        result = solver.check()

        if result == z3_lib.unsat:
            return False, None, "unsat"

        if result == z3_lib.unknown:
            return False, None, "unknown"

        model = solver.model()

        try:
            ans = model.eval(
                z3_lib.Const("answer", z3_lib.RealSort()),
                model_completion=True
            )
            return True, str(ans), "sat"
        except:
            return True, None, "sat_no_answer"

    except Exception as e:
        return False, None, f"error:{str(e)[:80]}"


# ============================================================
# COMPATIBILITY WRAPPER (FIX IMPORT ERROR)
# ============================================================

def validate_mutation(smt_code: str):
    return validate_smt(smt_code)