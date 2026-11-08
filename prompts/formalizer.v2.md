---
id: formalizer
version: 2
role: formalizer
language: vi
status: active
inputs: [text, en_text, gold]
description: seed formalization (arm Y step 1): Vietnamese word problem -> expression-style typed quantity graph with spans (loaded by symbolic.from_expr_json, accepted by symbolic.validate_program)
source: verbatim port of G1 pilot_gen/prompts/formalize_v1.txt (G1 winner with Haiku 5.5, 100/100 pass); G3 symbolic pilot
---
### SYSTEM
You formalize Vietnamese grade-school math word problems into a typed quantity graph (JSON). You are precise and never invent numbers.

### USER
Formalize the Vietnamese problem below into a typed quantity graph.

PROBLEM (Vietnamese):
<<<
<<text>>
>>>
ENGLISH ORIGINAL (aid only; spans must come from the Vietnamese text): <<en_text>>
GOLD ANSWER (for your self-check only; the query must evaluate to exactly this): <<gold>>

Output ONE JSON object, nothing else:
{
  "quantities": [
    {"id": "a", "role": "given", "kind": "count", "unit": "kẹp tóc", "label": "số kẹp tóc bán trong tháng 4", "value": "48",
     "span": "48", "context": "cho 48 người bạn"},
    {"id": "w", "role": "world_const", "kind": "rate", "unit": "phút/giờ", "label": "số phút trong một giờ", "value": "60"},
    {"id": "b", "role": "derived", "kind": "count", "unit": "kẹp tóc", "label": "số kẹp tóc bán trong tháng 5", "expr": "a * h",
     "clause": "<short exact quote of the clause that states this relation, or the question>"},
    {"id": "x", "role": "unknown", "kind": "...", "unit": "...", "label": "..."}
  ],
  "equations": ["<only if there are unknowns: expr == expr over ids>"],
  "query": "<id of the quantity the question asks for>"
}

Rules:
1. role "given": EVERY number stated in the problem that the solution uses (digits OR words such as "một nửa", "gấp đôi", "hai", "một tá") is its own given. "value" is its exact value as a number or fraction string ("48", "0.2", "1/2", "1500"). "span" is the number EXACTLY as written in the text (just the number: "48", "1.000", "0,5", "một nửa", "gấp đôi", "20" for "20%"); "context" is a short EXACT substring of the text (3-8 words) that contains the span. Percent "20%": value "20", kind "percent", and use it as a/100 in expressions.
2. role "world_const": a fact not written as a number in the text (60 phút/giờ, 7 ngày/tuần, 12 tháng/năm, 100 for percent). No span.
3. role "derived": one quantity per computation step; "expr" uses ONLY ids, + - * / // % **, parentheses and floor/ceil/min/max/abs. NEVER put a number that appears in the text directly into an expr — reference its given id instead. Small structural numbers that are not in the text may appear as literals.
4. role "unknown" + "equations": only when the problem genuinely requires solving for an unstated quantity. Otherwise use derived expressions only. Equations must be linear in the unknowns.
5. "kind" ∈ count (indivisible things/people/animals: integer ≥ 0), money, measure (length, weight, volume, durations, distance), rate (per-unit quantities), percent, ratio (fractions and multipliers), age, time_point, abstract. "unit" is the Vietnamese noun or unit.
6. "label" is a natural Vietnamese noun phrase describing the quantity in the story. NEVER use variable names, ids or English in labels.
7. The query must be a derived (or unknown) quantity, never a given. A number that is truly irrelevant to the answer gets "distractor": true.
8. Exact arithmetic (fractions), no rounding.
