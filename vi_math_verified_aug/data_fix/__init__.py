"""W3 eval/seed repair: build cleaned, versioned datasets under ``data/clean/``.

Never modifies anything under ``data/test``, ``data/translation`` or ``data/finetune``;
every output is a new file under ``data/clean/`` with a JSON datacard next to it.

Modules
-------
common          IO, datacards, LaTeX/number helpers shared by every builder
fetch_sources   download the public English references (GSM8K-Platinum, MATH-500, SVAMP, ASDiv)
nim_client      rate-limited (<=10 RPM), fully logged NIM client; refuses test-split ids
mta_vi          MTA-vi translation (masking + triggered glossary + G0/G1 gates), T4 §3
build_gsm8k_test, build_math_test, fix_gsm8k_train, fix_math_train, build_seed, fix_asdiv_svamp
"""
