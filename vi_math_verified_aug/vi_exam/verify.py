"""Vi-ExamMath key verification lives in ``symkey.py`` (SymPy keys, answer comparison) and ``build.key_status``
(3-key status: provisional solution key, SymPy key, blind Claude key; T5 §3.1 stage 2)."""
from .symkey import agree_free, build_key, compare  # noqa: F401
