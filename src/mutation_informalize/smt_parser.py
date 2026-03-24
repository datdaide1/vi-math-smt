"""
SMT-LIB Parser v2
Robust parser/rebuilder for:
- Phase1 Agentic Repair
- Phase2 Mutation
- Phase3 Informalization

Backward-compatible:
- giữ nguyên tên file
- giữ nguyên class/function names
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================
# Data Classes
# ============================================================

@dataclass
class SMTVariable:
    name: str
    sort: str


@dataclass
class SMTAssert:
    raw: str

    is_trivial: bool = False

    references_answer: bool = False

    is_answer_definition: bool = False

    expr_type: str = "unknown"


@dataclass
class SMTParsed:

    logic: str

    variables: List[SMTVariable] = field(
        default_factory=list
    )

    asserts: List[SMTAssert] = field(
        default_factory=list
    )

    answer_var: Optional[str] = None

    answer_expr: Optional[str] = None

    raw_code: str = ""

    comments: List[str] = field(
        default_factory=list
    )


# ============================================================
# Internal Helpers
# ============================================================

SOLVER_COMMANDS = [
    "check-sat",
    "get-model",
    "get-value",
    "exit",
]


def strip_solver_commands(code: str) -> str:

    lines = []

    for line in code.splitlines():

        stripped = line.strip()

        should_skip = False

        for cmd in SOLVER_COMMANDS:

            if stripped.startswith(f"({cmd}"):
                should_skip = True
                break

        if not should_skip:
            lines.append(line)

    return "\n".join(lines)


def extract_balanced_expressions(
    code: str,
    keyword: str,
):

    results = []

    pattern = f"({keyword}"

    start = 0

    while True:

        idx = code.find(pattern, start)

        if idx == -1:
            break

        depth = 0

        end = idx

        for i in range(idx, len(code)):

            if code[i] == "(":
                depth += 1

            elif code[i] == ")":
                depth -= 1

                if depth == 0:
                    end = i + 1
                    break

        expr = code[idx:end]

        results.append(expr)

        start = end

    return results


def classify_assert(raw: str):

    raw_compact = " ".join(
        raw.split()
    )

    # --------------------------------------------------------
    # trivial
    # --------------------------------------------------------

    trivial_patterns = [
        r'^\(assert \(= \w+ -?\d+(?:\.\d+)?\)\)$',
        r'^\(assert \(= \w+ \w+\)\)$',
    ]

    is_trivial = any(
        re.match(p, raw_compact)
        for p in trivial_patterns
    )

    # --------------------------------------------------------
    # answer reference
    # --------------------------------------------------------

    references_answer = bool(
        re.search(
            r'(?<![\w])answer(?![\w])',
            raw,
        )
    )

    # --------------------------------------------------------
    # answer definition
    # --------------------------------------------------------

    is_answer_definition = bool(
        re.match(
            r'^\(assert\s+\(=\s+answer\s+',
            raw.strip(),
            re.DOTALL,
        )
    )

    # --------------------------------------------------------
    # expr type
    # --------------------------------------------------------

    expr_type = "unknown"

    if re.search(r'\(\s*=', raw):
        expr_type = "equality"

    elif re.search(r'\(\s*(>|<|>=|<=)', raw):
        expr_type = "inequality"

    elif re.search(r'\(\s*(and|or|not|=>)', raw):
        expr_type = "logical"

    return (
        is_trivial,
        references_answer,
        is_answer_definition,
        expr_type,
    )


# ============================================================
# Main Parser
# ============================================================

def parse_smt(code: str) -> SMTParsed:
    """
    Parse SMT-LIB into structured object.
    """

    cleaned_code = strip_solver_commands(
        code
    )

    result = SMTParsed(
        logic="QF_LRA",
        raw_code=cleaned_code,
    )

    # ========================================================
    # comments
    # ========================================================

    for line in cleaned_code.splitlines():

        line = line.strip()

        if line.startswith(";"):
            result.comments.append(line)

    # ========================================================
    # logic
    # ========================================================

    m = re.search(
        r'\(set-logic\s+([^)]+)\)',
        cleaned_code,
    )

    if m:
        result.logic = m.group(1).strip()

    # ========================================================
    # variables
    # ========================================================

    declare_patterns = [
        r'\(declare-fun\s+(\w+)\s*\(\)\s*([^)]+)\)',
        r'\(declare-const\s+(\w+)\s+([^)]+)\)',
    ]

    for pattern in declare_patterns:

        for m in re.finditer(
            pattern,
            cleaned_code,
        ):

            name = m.group(1).strip()

            sort = m.group(2).strip()

            result.variables.append(
                SMTVariable(
                    name=name,
                    sort=sort,
                )
            )

            if name == "answer":
                result.answer_var = "answer"

    # ========================================================
    # asserts
    # ========================================================

    assert_exprs = extract_balanced_expressions(
        cleaned_code,
        "assert",
    )

    for raw in assert_exprs:

        (
            is_trivial,
            references_answer,
            is_answer_definition,
            expr_type,
        ) = classify_assert(raw)

        sa = SMTAssert(
            raw=raw,
            is_trivial=is_trivial,
            references_answer=references_answer,
            is_answer_definition=is_answer_definition,
            expr_type=expr_type,
        )

        result.asserts.append(sa)

        # ----------------------------------------------------
        # answer expression
        # ----------------------------------------------------

        if is_answer_definition:

            m = re.match(
                r'\(assert\s+\(=\s+answer\s+(.+)\)\)',
                raw.strip(),
                re.DOTALL,
            )

            if m:

                expr = m.group(1).strip()

                # If expr has extra closing parenthesis due to greedy matching or malformed spacing,
                # we only remove them if open_count < close_count.
                open_count = expr.count("(")
                close_count = expr.count(")")
                
                while expr.endswith(")") and open_count < close_count:
                    expr = expr[:-1]
                    close_count -= 1

                result.answer_expr = expr.strip()

    return result


# ============================================================
# Rebuilder
# ============================================================

def rebuild_smt(
    parsed: SMTParsed,
    extra_vars: List[SMTVariable] = None,
    extra_asserts: List[str] = None,
    new_answer_expr: str = None,
) -> str:

    lines = []

    # ========================================================
    # logic
    # ========================================================

    lines.append(
        f"(set-logic {parsed.logic})"
    )

    # ========================================================
    # variables
    # ========================================================

    declared = set()

    for v in parsed.variables:

        if v.name in declared:
            continue

        declared.add(v.name)

        lines.append(
            f"(declare-const {v.name} {v.sort})"
        )

    if extra_vars:

        for v in extra_vars:

            if v.name in declared:
                continue

            declared.add(v.name)

            lines.append(
                f"(declare-const {v.name} {v.sort})"
            )

    # ========================================================
    # asserts
    # ========================================================

    for a in parsed.asserts:

        if (
            a.is_answer_definition
            and new_answer_expr
        ):

            lines.append(
                f"(assert (= answer {new_answer_expr}))"
            )

        else:
            lines.append(a.raw)

    if extra_asserts:

        for ea in extra_asserts:

            lines.append(ea)

    # ========================================================
    # solver commands
    # ========================================================

    lines.append("(check-sat)")

    if parsed.answer_var:

        lines.append(
            "(get-value (answer))"
        )

    return "\n".join(lines)


# ============================================================
# Utility Functions
# ============================================================

def get_non_answer_vars(
    parsed: SMTParsed
) -> List[SMTVariable]:

    return [
        v
        for v in parsed.variables
        if v.name != "answer"
    ]


def get_constant_vars(
    parsed: SMTParsed
) -> List[Tuple[str, float]]:

    results = []

    for a in parsed.asserts:

        m = re.match(
            r'\(assert\s+\(=\s+(\w+)\s+(-?\d+(?:\.\d+)?)\)\)',
            a.raw.strip(),
        )

        if not m:
            continue

        name = m.group(1)

        if name == "answer":
            continue

        try:

            value = float(
                m.group(2)
            )

            results.append(
                (name, value)
            )

        except:
            pass

    return results


def get_computed_vars(
    parsed: SMTParsed
) -> List[str]:

    const_names = {
        name
        for name, _
        in get_constant_vars(parsed)
    }

    return [
        v.name
        for v in parsed.variables
        if (
            v.name != "answer"
            and v.name not in const_names
        )
    ]