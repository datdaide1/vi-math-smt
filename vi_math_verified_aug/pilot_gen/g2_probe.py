"""G2: probe the free Gemini quota of candidate judge models, one tiny call per (key, model).

python -m vi_math_verified_aug.pilot_gen.g2_probe [models...]
Logged to logs/api/G2/gemini.jsonl (tag=probe). On 429 the QuotaFailure details (quota id + value) are printed,
which is how the free-tier limits are read (no key is printed).
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

from ..common.apilog import ApiLogger
from ..common.env import read_env, secret_values, split_keys
from ..common.llm import GeminiProvider, LLMRequest

MODELS = ["gemini-3.1-pro-preview", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.1-flash-lite"]


def quota_info(raw) -> list:
    out = []
    try:
        for d in raw["error"].get("details", []):
            for v in d.get("violations", []) or []:
                out.append({"quotaId": v.get("quotaId"), "value": v.get("quotaValue")})
    except Exception:
        pass
    return out


async def main(models):
    env = read_env()
    keys = split_keys(env.get("GEMINI_API_KEYS"))
    prov = GeminiProvider(keys)
    log = ApiLogger("G2", secrets=secret_values(env))
    res = {}
    for m in models:
        for i in range(len(keys)):
            req = LLMRequest(model=m, messages=[{"role": "user", "content": "Tính 17 + 25. Chỉ trả lời số."}],
                             max_tokens=256, temperature=0.0, item_id="probe", split="dev", tag="probe")
            t0 = time.perf_counter()
            r = await prov.send(req, i)
            lat = int((time.perf_counter() - t0) * 1000)
            log.log({"provider": "gemini", "model_requested": m, "lane": f"k{i}", "status": r.status,
                     "ok": r.status == 200, "tag": "probe", "split": "dev", "item_id": "probe",
                     "request": {"messages": req.messages}, "response_text": r.text, "raw_response": r.raw,
                     "usage": r.usage, "latency_ms": lat, "cost_usd": 0.0, "error": r.error})
            q = quota_info(r.raw) if r.status != 200 else []
            res[f"{m}#k{i}"] = (r.status, (r.text or "").strip()[:20], lat, q)
            print(m, f"k{i}", r.status, repr((r.text or "").strip()[:20]), f"{lat}ms", json.dumps(q)[:300],
                  (r.error or "")[:160] if r.status != 200 and not q else "", flush=True)
    await prov.aclose()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or MODELS))
