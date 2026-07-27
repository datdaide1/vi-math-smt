# Dataset Sources, Licenses, and Citations

This document records the provenance and upstream terms of the four benchmark
datasets used by Vi-Math-SMT. It is an attribution and release checklist, not
legal advice. Verify the upstream pages again before publishing a new dataset
release because dataset cards and provider terms can change.

## Source Summary

| Dataset | Canonical source | Pipeline mirror | License |
| --- | --- | --- | --- |
| GSM8K | [OpenAI dataset card](https://huggingface.co/datasets/openai/gsm8k) | `openai/gsm8k`, configuration `main` | [MIT](https://huggingface.co/datasets/openai/gsm8k) |
| MATH | [hendrycks/math](https://github.com/hendrycks/math) | `hendrycks/competition_math` | [MIT](https://github.com/hendrycks/math/blob/main/LICENSE) |
| SVAMP | [arkilpatel/SVAMP](https://github.com/arkilpatel/SVAMP) | `ChilleD/SVAMP` | [MIT](https://github.com/arkilpatel/SVAMP/blob/main/LICENSE) |
| ASDiv | [chaochun/nlu-asdiv-dataset](https://github.com/chaochun/nlu-asdiv-dataset) | `EleutherAI/asdiv` | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

## Redistribution Checklist

When publishing original, translated, normalized, or augmented records:

1. Name every upstream dataset and link to its canonical source.
2. Include the paper citation for every source represented in the release.
3. Preserve the MIT copyright/license notices for GSM8K, MATH, and SVAMP.
4. Attribute ASDiv to the Natural Language Understanding Laboratory, Institute
   of Information Science, Academia Sinica.
5. Mark Vietnamese translation, schema normalization, SMT formalization,
   filtering, and symbolic augmentation as modifications made by this project.
6. Do not represent ASDiv or ASDiv-derived records as available for commercial
   use; CC BY-NC 4.0 permits sharing and adaptation for non-commercial purposes
   with attribution.
7. Do not apply a blanket MIT label to a combined release that includes
   ASDiv-derived records. Document license applicability per subset and use a
   conservative non-commercial release policy for the combined artifact.
8. Keep train/evaluation provenance fields where practical so downstream users
   can identify the source and applicable terms of each record.

The repository's source-code license, if one is added, governs only original
project code and documentation unless it explicitly says otherwise. It does not
override dataset licenses or third-party service terms.

## Citations

### GSM8K

```bibtex
@article{cobbe2021training,
  title   = {Training Verifiers to Solve Math Word Problems},
  author  = {Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and
             Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert,
             Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Rei and
             Hesse, Christopher and Schulman, John},
  journal = {arXiv preprint arXiv:2110.14168},
  year    = {2021},
  url     = {https://arxiv.org/abs/2110.14168}
}
```

### MATH

```bibtex
@article{hendrycks2021measuring,
  title   = {Measuring Mathematical Problem Solving With the MATH Dataset},
  author  = {Hendrycks, Dan and Burns, Collin and Kadavath, Saurav and Arora,
             Akul and Basart, Steven and Tang, Eric and Song, Dawn and
             Steinhardt, Jacob},
  journal = {arXiv preprint arXiv:2103.03874},
  year    = {2021},
  url     = {https://arxiv.org/abs/2103.03874}
}
```

### SVAMP

```bibtex
@inproceedings{patel2021nlp,
  title     = {Are NLP Models really able to Solve Simple Math Word Problems?},
  author    = {Patel, Arkil and Bhattamishra, Satwik and Goyal, Navin},
  booktitle = {Proceedings of NAACL-HLT 2021},
  pages     = {2080--2094},
  year      = {2021},
  doi       = {10.18653/v1/2021.naacl-main.168},
  url       = {https://aclanthology.org/2021.naacl-main.168}
}
```

### ASDiv

```bibtex
@inproceedings{miao2020diverse,
  title     = {A Diverse Corpus for Evaluating and Developing English Math
               Word Problem Solvers},
  author    = {Miao, Shen-yun and Liang, Chao-Chun and Su, Keh-Yih},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for
               Computational Linguistics},
  pages     = {975--984},
  year      = {2020},
  doi       = {10.18653/v1/2020.acl-main.92},
  url       = {https://aclanthology.org/2020.acl-main.92}
}
```

## External API Credentials

Vi-Math-SMT can call Hugging Face, Google Gemini, NVIDIA NIM, and an optional
legacy proxy. API credentials must never be committed, bundled with a dataset,
embedded in a model card, or copied into execution logs.

- Store local values only in `.env` or an operating-system secret manager.
- Keep `.env.example` limited to empty variable names and documentation.
- Use separate development and production credentials where possible.
- Restrict token scopes to the minimum required operation.
- Treat every prompt and uploaded record as data disclosed to the selected API
  provider; review its current retention, training, privacy, and billing terms.
- Revoke and rotate a credential immediately if it appears in Git history,
  notebook output, screenshots, issue text, or a shared archive.

API access does not grant rights to the input datasets and does not change any
upstream dataset license.
