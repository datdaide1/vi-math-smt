# Data content review — shared protocol (2026-11-07)

Goal: measure, by actually READING items, how good the existing data content is — both the
natural-language (Vietnamese) side and the SMT-LIB side — so keep/fix/discard decisions rest on evidence.

## Sampling
- Python `random.Random(20261107)`, sample WITHOUT replacement; stratify by `subject`/`level` (MATH) or
  uniformly (GSM8K-like). Default n = 150 items per file (report the exact n and the index list).
- Record the sampled indices so anyone can re-draw the same sample.

## Reading rule
- Each sampled item is read in full by the reviewer (you), alongside its English source where one exists
  (align by `index`). Regex/heuristic scans are allowed ONLY as a second, full-corpus pass to size a
  pattern you first saw by reading; never as a substitute for reading.
- For SMT items: also execute with Z3 (`.venv/Scripts/python`, z3 5.1.0 + sympy available), with a 10 s
  timeout per program.

## Error taxonomy (use these codes; add new ones as `X-<name>` if needed)
Natural language (Vietnamese):
- `NL-MEANING`  translation changes the math (wrong number, condition, quantity, question asked)
- `NL-OMIT`     a sentence/condition/clause dropped or added
- `NL-LATEX`    LaTeX/formula broken, altered, or mis-rendered
- `NL-FLUENCY`  unnatural / machine-like / wrong Vietnamese math terminology (not math-changing)
- `NL-LEAK`     artifacts leaked (variable names like h_123, SMT text, English left over, answer stated)
- `NL-UNANSW`   problem is unanswerable / ill-posed / ambiguous as written
- `NL-DUP`      duplicate or near-duplicate of another item
- `NL-TACKED`   (augmented) the "new" problem is the source story + an arithmetic tail bolted on with no
                story meaning, e.g. source "Lan có 2 kẹo, cho anh 1 cái, còn mấy cái?" → "…cho anh 1 cái
                và cộng thêm 0,5, Lan còn mấy cái?"
- `NL-ABSURD`   real-world nonsense: fractional/negative counts of discrete things (0,5 cái kẹo, −3 người),
                impossible rates/ages/prices, units that don't fit the story
- `NL-NEARDUP`  (augmented) same scenario/structure as its source item, only numbers/tail changed — adds no
                new reasoning pattern
**USER PRIORITY (2026-11-07):** the core problem is the *semantic content of the problem statement* — whether
the augmented problem is a meaningful, natural, coherent math problem a teacher would write — NOT counts,
sources, or solutions. For augmented data, always pair each item with its SOURCE problem and judge this first.
Answer / solution:
- `ANS-WRONG`   gold answer is wrong for the problem as written
- `ANS-FORMAT`  answer correct but in a form an auto-grader will mis-score (units, "x=", decimal comma, ugly float)
- `SOL-WRONG`   solution has a math error or does not reach the gold answer
- `SOL-INCONS`  solution inconsistent with the problem text (solves a different problem)
SMT:
- `SMT-UNFAITHFUL` program does not encode the problem as stated (wrong constraint/number/relation)
- `SMT-HARDCODE`   answer asserted as a literal rather than derived
- `SMT-NONUNIQUE`  constraints admit multiple answers / answer var under-constrained
- `SMT-ERROR`      parse/solver error, timeout, unsat
- `SMT-MISMATCH`   Z3 value ≠ gold answer
- `SMT-TRIVIAL`    program is a restatement of arithmetic with no modelling (e.g. answer = 3*4)

Severity: `critical` (item would teach/score something false) / `major` (usable after fix) / `minor`.

## Outputs per reviewer
1. `annotations/<file-tag>.jsonl`: one row per sampled item:
   `{"file":..., "index":..., "codes":[...], "severity":..., "note":"<1–2 sentence evidence, quote the offending span>"}`
   (rows with no issue: `"codes":[]`).
2. Report `NN_<tag>.md`: n, error rate per code with **Wilson 95% CI**, % items usable as-is / fixable /
   discard, 5–10 quoted worst examples (index + short quote), any full-corpus pattern sizing, and a
   verdict: KEEP / FIX (how, cost) / DISCARD for that file, with justification.
Do not modify any data file. Do not call paid APIs.
