# Light test-suite report

Generated 2026-11-08 09:36:53 local by `handoff/tools/run_light_tests.py` (one pytest process per file, per-process memory cap 300 MB via a Windows Job Object; peak = committed bytes).

**Totals:** 1 failed, 931 passed, 2 skipped, 2 xfailed

| File | Interpreter | Exit | Peak MB | Seconds | pytest summary |
|---|---|---|---|---|---|
| `test_analysis.py` | .venv | 0 | 95.2 | 2.3 | 18 passed in 1.55s |
| `test_analysis_heavy.py` | – | excluded | – | 0 | excluded (heavy: Modal only) |
| `test_arm_y_export.py` | .venv | 0 | 74.4 | 2.2 | 2 passed in 1.54s |
| `test_c3_attrib.py` | .venv | 0 | 55.2 | 1.0 | 7 passed in 0.43s |
| `test_c3_d5_audit.py` | .venv | 0 | 45.2 | 0.9 | 2 passed in 0.36s |
| `test_c3_r1.py` | .venv | 0 | 95.5 | 1.4 | 1 passed in 0.82s |
| `test_c3_stats.py` | .venv | 0 | 39.1 | 1.0 | 7 passed in 0.50s |
| `test_common_apilog.py` | .venv | 0 | 34.1 | 0.9 | 4 passed, 1 skipped in 0.35s |
| `test_common_jobs.py` | .venv | 0 | 31.8 | 0.8 | 7 passed in 0.32s |
| `test_common_llm.py` | .venv | 0 | 37.2 | 1.5 | 28 passed in 0.93s |
| `test_common_prompts_config.py` | .venv | 1 | 32.9 | 0.7 | 1 failed, 18 passed in 0.23s |
| `test_common_ratelimit.py` | .venv | 0 | 31.0 | 0.6 | 11 passed in 0.08s |
| `test_decontam.py` | .venv | 0 | 43.0 | 0.8 | 39 passed in 0.32s |
| `test_handoff_tools.py` | .venv | 0 | 30.8 | 0.6 | 5 passed in 0.11s |
| `test_math_verify.py` | .venv | 0 | 80.9 | 7.9 | 368 passed, 1 skipped, 2 xfailed in 7.31s |
| `test_pilot_symbolic.py` | .venv | 0 | 71.3 | 1.9 | 3 passed in 1.19s |
| `test_pilot_symbolic_v2.py` | .venv | 0 | 73.1 | 1.6 | 8 passed in 0.91s |
| `test_qa_training_audit.py` | .venv | 0 | 77.8 | 2.4 | 16 passed in 1.87s |
| `test_scale_gen.py` | .venv | 0 | 35.8 | 0.7 | 16 passed in 0.22s |
| `test_seeds.py` | .venv | 0 | 30.8 | 0.5 | 8 passed in 0.06s |
| `test_select_v1.py` | .venv | 0 | 73.7 | 2.4 | 12 passed, 2 warnings in 1.91s |
| `test_symbolic_g3_fixes.py` | .venv | 0 | 72.8 | 1.8 | 32 passed in 1.18s |
| `test_symbolic_ir.py` | .venv | 0 | 93.6 | 1.6 | 79 passed in 1.00s |
| `test_symbolic_numerals.py` | .venv | 0 | 69.0 | 1.2 | 51 passed in 0.65s |
| `test_symbolic_operators.py` | .venv | 0 | 136.2 | 19.6 | 53 passed in 18.64s |
| `test_symbolic_rules.py` | .venv | 0 | 96.1 | 7.6 | 27 passed in 6.87s |
| `test_train_cell.py` | alt: E:/venv_y1/Scripts/python.exe | 0 | 188.6 | 6.7 | 3 passed in 5.13s |
| `test_train_core.py` | alt: E:/venv_y1/Scripts/python.exe | 0 | 35.3 | 0.7 | 23 passed in 0.14s |
| `test_train_evalrun.py` | alt: E:/venv_y1/Scripts/python.exe | 0 | 189.9 | 3.1 | 9 passed in 2.07s |
| `test_train_grpo.py` | alt: E:/venv_y1/Scripts/python.exe | 0 | 36.5 | 0.8 | 10 passed in 0.20s |
| `test_train_guards.py` | alt: E:/venv_y1/Scripts/python.exe | 0 | 189.1 | 2.8 | 17 passed in 1.87s |
| `test_train_kaggle.py` | alt: E:/venv_y1/Scripts/python.exe | 0 | 34.9 | 1.0 | 10 passed in 0.34s |
| `test_train_launcher_ready.py` | .venv | 0 | 35.2 | 0.7 | 8 passed in 0.16s |
| `test_train_sft_loop.py` | alt: E:/venv_y1/Scripts/python.exe | 0 | 188.9 | 3.2 | 9 passed in 2.21s |
| `test_uniqueness.py` | .venv | 0 | 50.8 | 0.6 | 10 passed in 0.10s |
| `test_vi_exam_qa.py` | .venv | 0 | 66.8 | 1.4 | 5 passed in 0.80s |
| `test_vi_exam_segment.py` | .venv | 0 | 46.9 | 1.0 | 5 passed in 0.47s |

## Output of files that did not exit 0

### test_common_prompts_config.py (exit 1)

```
F..................                                                      [100%]
================================== FAILURES ===================================
__________________ test_repo_registry_has_all_role_templates __________________

    def test_repo_registry_has_all_role_templates():
        reg = PromptRegistry()
        roles = {p.role for p in reg.all()}
        assert ROLES_REQUIRED <= roles
        for p in reg.all():
>           assert p.language == "vi" and p.version >= 1 and len(p.sha256) == 64
E           AssertionError: assert ('en' == 'vi'
E             
E             - vi
E             + en)

tests\test_common_prompts_config.py:39: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_common_prompts_config.py::test_repo_registry_has_all_role_templates
1 failed, 18 passed in 0.23s
```

