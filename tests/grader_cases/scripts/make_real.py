"""Emit tests/grader_cases/real_vi_gsm8k.jsonl: real Qwen3/Qwen3.5 base-model outputs on Vi-GSM8K test
(jobs/outputs/w2-pilot-v1), hand-labelled by X1 by reading each output BEFORE running the new grader.
Label = "does the response's committed final answer equal the gold under the response's own number
convention?"  A response with no committed final answer (truncated repetition loop) is labelled False."""
import json, glob, os, re, sys

M = {"06": "Qwen3-0.6B-Base", "17": "Qwen3-1.7B-Base", "08": "Qwen3.5-0.8B-Base", "2B": "Qwen3.5-2B-Base"}
# (model, test_index): (label, reason)
L = {
    ("06", 5): (False, "137.6 vs 64"),
    ("06", 70): (False, "20475 vs 7425; body has $20,475 (en grouping)"),
    ("06", 265): (False, "290000 vs 9500; body uses space grouping 350 000"),
    ("06", 332): (True, "44; body has 0,4 decimal commas"),
    ("06", 361): (False, "0.0375 vs 20"),
    ("06", 374): (False, "15 vs 45; body 7,5"),
    ("06", 514): (True, "boxed 90.000 = 90000 (vi grouping, body 120.000 / 30.000)"),
    ("06", 582): (False, "109.271.428 vs 129200"),
    ("06", 629): (False, "100.000 = 100000 vs 17500"),
    ("06", 829): (False, "12500 vs 14000"),
    ("06", 913): (False, "10 vs 6"),
    ("06", 1014): (False, "-6,75 vs 3"),
    ("06", 1018): (False, "19 1/3 vs 60"),
    ("06", 1103): (False, "74 vs 90"),
    ("06", 1285): (True, "1218; body 1.360 - 142 = 1.218"),
    ("17", 361): (False, "48 vs 20"),
    ("17", 436): (True, "15; body 2,5"),
    ("17", 514): (True, "90000"),
    ("17", 582): (True, "129200"),
    ("17", 1179): (False, "23.33% vs 50"),
    ("08", 226): (False, "107 vs 33"),
    ("08", 361): (False, "30.000 vs 20 (either reading)"),
    ("08", 696): (False, "238,333 vs 145"),
    ("08", 715): (True, "385000; body $385,000"),
    ("08", 806): (False, "0,0024 vs 70"),
    ("08", 829): (False, "6.000 = 6000 vs 14000"),
    ("08", 1285): (True, "1218"),
    ("08", 1318): (False, "18 vs 14"),
    ("2B", 226): (True, "33; body 33,33%"),
    ("2B", 629): (False, "332.500 vs 17500"),
    ("2B", 728): (False, "100% vs 40"),
    ("2B", 806): (False, "0,725 vs 70"),
    ("2B", 829): (True, "boxed 14.000 = 14000 (vi grouping, body 50.000-36.000=14.000)"),
    ("2B", 1199): (False, "2,25 vs 2: decimal comma, not 2 and not 225"),
    ("08", 101): (False, "9,75 vs 6"),
    ("08", 475): (False, "5,25 vs 20"),
    ("2B", 46): (False, "23,26 vs 163"),
    ("06", 999): (False, "3/8 vs 25"),
    ("08", 954): (False, "-8/3 vs 34"),
    ("17", 153): (False, "2/3 vs 48"),
    # random plain boxed integers
    ("17", 485): (True, "220"), ("17", 472): (True, "6"), ("17", 425): (False, "180 vs 150"),
    ("06", 206): (True, "860"), ("2B", 132): (True, "60"), ("06", 1075): (False, "30 vs 16"),
    ("17", 403): (True, "81"), ("2B", 327): (True, "31"), ("06", 1317): (False, "10 vs 5"),
    ("08", 343): (True, "180000"), ("06", 389): (False, "11 vs 6"),
    ("08", 1301): (False, "39 vs 23 (23 appears earlier in the body)"),
    ("2B", 509): (False, "5 vs 4"), ("17", 1162): (False, "8 vs 3"), ("06", 243): (True, "250"),
    ("2B", 1081): (True, "195"), ("06", 690): (False, "3360 vs 5760"), ("08", 663): (False, "17593 vs 193"),
}
rows = {}
for f in sorted(glob.glob("jobs/outputs/w2-pilot-v1/results/*/preds.jsonl")):
    model = os.path.basename(os.path.dirname(f))
    for l in open(f, encoding="utf-8"):
        d = json.loads(l)
        rows[(model, d["test_index"])] = d
out = []
for (mk, ti), (lab, why) in L.items():
    d = rows[(M[mk], ti)]
    out.append((M[mk], ti, d, lab, why))
# every unboxed output: all are truncated generations (repetition loops / cut mid-sentence), no committed answer
for (model, ti), d in sorted(rows.items()):
    if "\\boxed" not in d["output"]:
        out.append((model, ti, d, False, "no \\boxed, no cue: truncated generation (loop / cut), no committed answer"))
seen = set()
with open(sys.argv[1], "w", encoding="utf-8") as fo:
    for model, ti, d, lab, why in out:
        cid = f"R-{model}-{ti}"
        assert cid not in seen, cid
        seen.add(cid)
        assert re.fullmatch(r"-?\d+", d["gold"]), d["gold"]
        fo.write(json.dumps({"id": cid, "pred": d["output"],
                             "gold": {"answer_type": "int", "answer_canonical": d["gold"], "locale": "mixed"},
                             "expect": lab, "note": why, "old_scorer": d["correct"],
                             "source": f"real:jobs/outputs/w2-pilot-v1/results/{model}/preds.jsonl#test_index={ti}"},
                            ensure_ascii=False) + "\n")
print(len(seen), "real cases")
