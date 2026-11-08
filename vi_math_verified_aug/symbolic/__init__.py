"""Symbolic core library for arm A (T1 §2.1-§2.2, §2.5; 01_TECHNICAL_DESIGN §3.1-§3.4). Pure code, no API calls.

    units      QType, Unit, Vietnamese unit/classifier lexicon, type grids
    numerals   Vietnamese numeral normalizer / formatter
    ir         Quantity, Op, Equation, Program (+ versioned JSON), ProgramBuilder, from_expr_json (G1 format)
    executor   exact executor, type/unit propagation, constraints
    validate   formalizer acceptance (i)-(v)
    rules      is_answer_tail, is_identity, clean_answer, plausible, unique_solution, gate
    operators  M1-M6 -> (Program, [TextEdit])
    render     TextEdit, apply_substitutions, tagged text, tag_multiset_ok
    legacy_ops the old answer-expression operators re-expressed in the IR (C3 baseline)
    fixtures   hand-written IRs for 23 Vi-GSM8K + 5 Vi-MATH algebra seeds
"""
from .units import QType, Unit, resolve_unit
from .numerals import parse_literal, parse_number, fmt_vi, fmt_like, contains_value
from .ir import SCHEMA_VERSION, IRError, Quantity, Op, Equation, Program, ProgramBuilder, from_expr_json
from .executor import ExecError, execute, run, check, type_violations, infer_types
from .validate import Violation, validate_program, accepted
from .rules import (is_answer_tail, is_identity, clean_answer, answer_issues, plausible, unique_solution,
                    is_semantic_tail, gate)
from .render import (TextEdit, apply_substitutions, render_substitutions, tagged_text, parse_tags, strip_tags,
                     tag_multiset_ok, tag_issues, leaks_answer)
from .operators import (NotApplicable, MutationResult, OPERATORS, mutate, m1_resample, m2_inverse,
                        m3_insert_step, m4_relation_change, m5_query_shift, m6_compose)

__all__ = [n for n in dir() if not n.startswith("_")]
