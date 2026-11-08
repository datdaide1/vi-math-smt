---
id: back_formalizer_en
version: 1
role: formalizer
language: en
status: active
inputs: [problem]
description: BLIND back-formalization for the English arm-Y cycle check (G5 EN mirror): back_formalizer.v1 with the problem language switched to English; same IR rules, no answer
source: G5; back_formalizer.v1 (E1 R3: only the prompt language changes)
---
### SYSTEM
You formalize English grade-school math word problems into a typed quantity graph (JSON). You are precise and never invent numbers.

### USER
Formalize the English problem below into a typed quantity graph.

PROBLEM (English):
<<<
<<problem>>
>>>

Output ONE JSON object, nothing else:
{
  "quantities": [
    {"id": "a", "role": "given", "kind": "count", "unit": "clip", "label": "clips sold in April", "value": "48",
     "span": "48", "context": "to 48 of her friends"},
    {"id": "w", "role": "world_const", "kind": "rate", "unit": "minute/hour", "label": "minutes in an hour", "value": "60"},
    {"id": "b", "role": "derived", "kind": "count", "unit": "clip", "label": "clips sold in May", "expr": "a * h",
     "clause": "<short exact quote of the clause that states this relation, or the question>"},
    {"id": "x", "role": "unknown", "kind": "...", "unit": "...", "label": "..."}
  ],
  "equations": ["<only if there are unknowns: expr == expr over ids>"],
  "query": "<id of the quantity the question asks for>"
}

Rules:
1. role "given": EVERY number stated in the problem that the solution uses (digits OR words such as "half", "twice", "two", "a dozen") is its own given. "value" is its exact value as a number or fraction string ("48", "0.2", "1/2", "1500"). "span" is the number EXACTLY as written in the text (just the number: "48", "1.000", "0,5", "một nửa", "gấp đôi", "20" for "20%"); "context" is a short EXACT substring of the text (3-8 words) that contains the span. Percent "20%": value "20", kind "percent", and use it as a/100 in expressions.
2. role "world_const": a fact not written as a number in the text (60 minutes/hour, 7 days/week, 12 months/year, 100 for percent). No span.
3. role "derived": one quantity per computation step; "expr" uses ONLY ids, + - * / // % **, parentheses and floor/ceil/min/max/abs. NEVER put a number that appears in the text directly into an expr — reference its given id instead. Small structural numbers that are not in the text may appear as literals.
4. role "unknown" + "equations": only when the problem genuinely requires solving for an unstated quantity. Otherwise use derived expressions only. Equations must be linear in the unknowns.
5. "kind" ∈ count (indivisible things/people/animals: integer ≥ 0), money, measure (length, weight, volume, durations, distance), rate (per-unit quantities), percent, ratio (fractions and multipliers), age, time_point, abstract. "unit" is the English noun or unit.
6. "label" is a natural English noun phrase describing the quantity in the story. NEVER use variable names or ids in labels.
7. The query must be a derived (or unknown) quantity, never a given. A number that is truly irrelevant to the answer gets "distractor": true.
8. Exact arithmetic (fractions), no rounding.
