"""C3: cause coding of every NL-ABSURD item in G3 and G4 (E, R1, R2, N), by reading the Opus note, the item text
and the paired E / N / seed verdicts on the same slot (D5, 2026-11-07; PROTOCOL reading rule).

Cause codes (first = primary, the defect the Opus note leads with or the one that already exists upstream):
  SEED       the untransformed seed already has the same defect (seed judged NL-ABSURD for the same reason)  -> (i)
  OP-VALUE   a value chosen by the operator is implausible (M1 resample, M2 stated total, M4 factor)        -> (ii)
  OP-STRUCT  the operator's structural edit has no story meaning (M3 template misfit, M4 flip, M6 bridge)    -> (ii)
  IR         the formalizer's IR mis-encodes the seed (redundant given encoded as a distractor; G3 fix 5)   -> formalizer
  LOC-PRICE  USD -> dong localization at 1 USD = 1.000 dong gives implausibly low VND prices (G4 R1/R2)       -> (iii)
  LOC-UNIT   imperial unit relabelled to a metric unit without rescaling (ounce -> gam/lit)                  -> (iii)
  REN-STORY  the renderer changed the setting / dropped a condition so the numbers no longer fit the story    -> (iii)
  GEN        the free generator (arm N) chose the implausible scenario or step itself                         -> (iv)
"""
from __future__ import annotations

FACTOR = {"SEED": "(i) seed", "OP-VALUE": "(ii) operator", "OP-STRUCT": "(ii) operator", "IR": "formalizer (IR)",
          "LOC-PRICE": "(iii) renderer/localize", "LOC-UNIT": "(iii) renderer/localize",
          "REN-STORY": "(iii) renderer/localize", "GEN": "(iv) generator"}

# key = "<variant>:<slot>"; value = cause codes, primary first
CODING: dict[str, list[str]] = {
    # ---- G3 (editor.v3 renderer, G3 operators incl. M6); 30 items
    "G3:2ca4d7b06e60:2": ["OP-STRUCT"], "G3:203d8a2add9c:3": ["OP-STRUCT"], "G3:22968fe49dd0:3": ["OP-VALUE"],
    "G3:40d6ab565c80:1": ["OP-STRUCT"], "G3:40d6ab565c80:2": ["OP-VALUE"], "G3:316a32678874:3": ["OP-STRUCT"],
    "G3:50502de35d64:1": ["OP-STRUCT"], "G3:40d6ab565c80:3": ["OP-STRUCT"], "G3:50502de35d64:2": ["OP-STRUCT"],
    "G3:3e3d54aae6e4:2": ["OP-STRUCT"], "G3:5c8a3139f914:3": ["OP-VALUE"], "G3:1bf27fde2604:2": ["OP-VALUE"],
    "G3:53512058c9b4:1": ["OP-STRUCT"], "G3:512147291e68:1": ["SEED"], "G3:5c8a3139f914:2": ["OP-STRUCT"],
    "G3:5e98d9f5dcfc:3": ["OP-VALUE"], "G3:5e98d9f5dcfc:2": ["OP-STRUCT"], "G3:512147291e68:2": ["SEED", "OP-STRUCT"],
    "G3:795cff975970:2": ["OP-STRUCT"], "G3:9fc4ec0a02e8:2": ["OP-STRUCT"], "G3:a2924f2bbf54:2": ["OP-STRUCT"],
    "G3:bc5df8fb8ac0:1": ["OP-STRUCT"], "G3:c3bf7d6bd57c:1": ["SEED", "OP-STRUCT"], "G3:bc4b7eefd798:3": ["OP-STRUCT"],
    "G3:d46a5281e278:1": ["SEED", "OP-VALUE"], "G3:dd6e61a413a4:3": ["OP-VALUE"], "G3:ddf2b33b9244:2": ["SEED"],
    "G3:ddf2b33b9244:1": ["SEED", "OP-VALUE"], "G3:d46a5281e278:3": ["SEED", "OP-STRUCT"],
    "G3:df5a7d0912c8:2": ["OP-STRUCT"],
    # ---- G4 E (same editor, fixed operators); 10 items, all identical programs/texts to G3
    "E:1bf27fde2604:2": ["OP-VALUE"], "E:22968fe49dd0:3": ["OP-VALUE"], "E:40d6ab565c80:2": ["OP-VALUE"],
    "E:512147291e68:1": ["SEED"], "E:5c8a3139f914:3": ["OP-VALUE"], "E:5e98d9f5dcfc:3": ["OP-VALUE"],
    "E:9fc4ec0a02e8:2": ["OP-STRUCT"], "E:d46a5281e278:1": ["SEED", "OP-VALUE"], "E:dd6e61a413a4:3": ["OP-VALUE"],
    "E:df5a7d0912c8:2": ["OP-STRUCT"],
    # ---- G4 R1 (E + fluency pass, 1 USD = 1.000 dong); 24 items
    "R1:1bf27fde2604:2": ["OP-VALUE", "LOC-PRICE"], "R1:1bf27fde2604:3": ["LOC-PRICE", "OP-STRUCT"],
    "R1:203d8a2add9c:1": ["LOC-UNIT"], "R1:203d8a2add9c:2": ["LOC-UNIT"], "R1:203d8a2add9c:3": ["LOC-UNIT"],
    "R1:22968fe49dd0:3": ["OP-VALUE"], "R1:2d00a50d7654:1": ["REN-STORY"],
    "R1:3e3d54aae6e4:2": ["LOC-PRICE", "REN-STORY"], "R1:40d6ab565c80:2": ["OP-VALUE"], "R1:512147291e68:1": ["SEED"],
    "R1:53512058c9b4:3": ["LOC-PRICE", "REN-STORY"], "R1:5c8a3139f914:3": ["OP-VALUE"],
    "R1:5e98d9f5dcfc:1": ["LOC-PRICE", "OP-STRUCT"], "R1:5e98d9f5dcfc:3": ["OP-VALUE", "LOC-PRICE"],
    "R1:6af8612c2274:1": ["LOC-PRICE"], "R1:b67b6408ec38:1": ["LOC-PRICE"], "R1:b67b6408ec38:3": ["LOC-PRICE"],
    "R1:c8357de93a24:2": ["IR", "LOC-PRICE"], "R1:c8357de93a24:3": ["IR", "LOC-PRICE"],
    "R1:d46a5281e278:1": ["SEED", "OP-VALUE"], "R1:dd6e61a413a4:3": ["OP-VALUE"],
    "R1:df5a7d0912c8:2": ["OP-STRUCT", "LOC-PRICE"], "R1:f991c76e758c:1": ["LOC-PRICE"], "R1:f991c76e758c:3": ["LOC-PRICE"],
    # ---- G4 R2 (free render from the localized IR); 22 items
    "R2:1bf27fde2604:2": ["LOC-PRICE", "OP-VALUE"], "R2:1bf27fde2604:3": ["LOC-PRICE", "OP-STRUCT"],
    "R2:203d8a2add9c:1": ["LOC-UNIT", "REN-STORY"], "R2:203d8a2add9c:3": ["LOC-UNIT"],
    "R2:22968fe49dd0:3": ["OP-VALUE", "REN-STORY"], "R2:2ca4d7b06e60:1": ["REN-STORY"], "R2:316a32678874:2": ["REN-STORY"],
    "R2:3e3d54aae6e4:2": ["LOC-PRICE"], "R2:5c8a3139f914:3": ["OP-VALUE", "LOC-UNIT"],
    "R2:5e98d9f5dcfc:1": ["LOC-PRICE", "REN-STORY"], "R2:5e98d9f5dcfc:3": ["OP-VALUE", "LOC-PRICE"],
    "R2:68d313f51190:1": ["REN-STORY", "OP-VALUE"], "R2:6af8612c2274:1": ["LOC-PRICE"], "R2:6af8612c2274:2": ["LOC-PRICE"],
    "R2:9fc4ec0a02e8:3": ["REN-STORY", "OP-VALUE"], "R2:b67b6408ec38:1": ["LOC-PRICE"],
    "R2:b67b6408ec38:3": ["LOC-PRICE", "REN-STORY"], "R2:c8357de93a24:2": ["IR", "LOC-PRICE"],
    "R2:c8357de93a24:3": ["IR", "LOC-PRICE"], "R2:dd6e61a413a4:3": ["OP-VALUE"], "R2:f991c76e758c:1": ["LOC-PRICE"],
    "R2:f991c76e758c:3": ["LOC-PRICE"],
    # ---- R1b (fluency.v2, 1 USD = 10.000 dong, no imperial relabel; judged by D5); 13 items
    "R1b:1bf27fde2604:2": ["OP-VALUE"], "R1b:1bf27fde2604:3": ["OP-STRUCT"], "R1b:22968fe49dd0:3": ["OP-VALUE"],
    "R1b:2d00a50d7654:1": ["REN-STORY"], "R1b:40d6ab565c80:2": ["OP-VALUE"], "R1b:512147291e68:1": ["SEED"],
    "R1b:5c8a3139f914:3": ["OP-VALUE"], "R1b:5e98d9f5dcfc:3": ["OP-VALUE"], "R1b:c860eb7e961c:2": ["LOC-PRICE", "SEED"],
    "R1b:d46a5281e278:1": ["SEED", "OP-VALUE", "REN-STORY"], "R1b:dd6e61a413a4:3": ["OP-VALUE"],
    "R1b:df5a7d0912c8:1": ["LOC-PRICE"], "R1b:df5a7d0912c8:2": ["OP-STRUCT", "LOC-PRICE"],
    # ---- G4 N (Haiku native_generator.v3, same slots); 5 items
    "N:5c8a3139f914:3": ["GEN"], "N:dd6e61a413a4:2": ["GEN"], "N:ddf2b33b9244:3": ["SEED", "GEN"],
    "N:df5a7d0912c8:2": ["GEN"], "N:df5a7d0912c8:3": ["GEN"],
}


def summarize(items_by_variant: dict[str, list]) -> dict:
    """Per variant: absurd items, primary-cause counts, any-cause counts, factor shares, and the counterfactual
    absurd rate if every LOC-* (localization) cause were removed (items whose only causes are LOC-*)."""
    out = {}
    for v, items in items_by_variant.items():
        absurd = [i for i in items if i.y["absurd"]]
        coded = [f"{v}:{i.slot}" for i in absurd]
        missing = [k for k in coded if k not in CODING]
        prim, anyc, fac_prim = {}, {}, {}
        for k in coded:
            cs = CODING.get(k, ["UNCODED"])
            prim[cs[0]] = prim.get(cs[0], 0) + 1
            f = FACTOR.get(cs[0], "uncoded")
            fac_prim[f] = fac_prim.get(f, 0) + 1
            for c in set(cs):
                anyc[c] = anyc.get(c, 0) + 1
        only_loc = sum(all(c.startswith("LOC") for c in CODING.get(k, ["X"])) for k in coded)
        out[v] = {"n_items": len(items), "n_absurd": len(absurd), "primary": prim, "any": anyc,
                  "factor_primary": fac_prim, "missing_coding": missing,
                  "absurd_without_loc_only": len(absurd) - only_loc}
    return out
