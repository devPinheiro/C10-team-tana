# C10 Team Tana: Domain-Specific SLM for Medical Advice

TRI AI Saturdays, Cohort 10. A retrieval-then-generate small language model that answers patient-facing clinical guidance questions in 1–3 evidence-grounded sentences, built for low-resource and frontline health settings. Educational benchmark only, not for clinical use.

**Track:** Domain-Specific SLM for Medical Advice (Kaggle).

All four Cohort Challenges are in `[docs/](docs/)`.

## Dataset

Training data combines three sources, documented in `[docs/data_card.pdf](docs/data_card.pdf)`:

1. **Competition corpus** (`data/documents.csv`, `data/original_train_qa.csv`) — 24 public-health factsheets (~36 words each) and 43 question–answer pairs. Eight topics (chronic disease, infectious disease, maternal health, medication safety, nutrition, vaccination, emergency triage, mental-health basics), each with care-setting and population labels.
2. **Team-authored augmentation** — 98 additional pairs written by Team Tana, each linked to an existing `document_id` and checked against a shared rubric: answer fully traceable to the factsheet, 1–3 sentences, direct verdict, no hedging, no external medical knowledge.
3. **Externally sourced pairs** (`data/train_qa_all.csv`, `data/documents_all.csv`) — 60 pairs grounded in paraphrased WHO, UNICEF, and CDC guidance, with `source_url` citations.

The final training merge is `data/combined_train_qa.csv` (201 pairs) and `data/combined_documents.csv` (35 factsheets). All content is synthetic educational text or paraphrased public guidance — no real patient records, and no copyrighted text reproduced verbatim.

## Training Pipeline

**Collection and preprocessing.** Gold training joins each question to its factsheet by `document_id`. Test questions have no gold document, so `scripts/retrieval_step.py` first filters by `topic` / `care_setting` / `population`, then ranks remaining factsheets with `all-MiniLM-L6-v2` cosine similarity. A four-stage fallback (exact metadata → topic+population → topic → full corpus) guarantees every question gets a context. Scores below 0.4 are flagged for manual review.

**Hyperparameter search.** We compared a retrieval-only baseline with two Qwen fine-tunes (91 vs 141 pairs), then moved to Gemma-2-2B-IT on the full 201-pair merge. Final settings: LoRA rank 16, 6 epochs, learning rate 2e-4, cosine schedule, warmup 5, batch 2 × 4 accumulation, max length 512, gradient checkpointing off.

**Model and design choices.** `scripts/finetune_clinical_slm.py` fine-tunes `google/gemma-2-2b-it` with QLoRA (4-bit NF4, LoRA on `q/k/v/o_proj`). The prompt is fixed: instruction + retrieved context + question + `Answer:`. Generation is greedy (`do_sample=False`); post-processing strips hedge openers and caps answers at three sentences.

We chose QLoRA over full fine-tuning to limit overfitting on 201 examples, and retrieval-then-generate over pure generation so every answer stays traceable to a source document (safety, groundedness, and accessibility from Challenge 1).

## Evaluation

The competition metric is mean Levenshtein distance against 11 hidden-reference test questions (lower is better):

| Iteration                         | Training pairs          | Score |
| --------------------------------- | ----------------------- | ----- |
| Retrieval baseline (no fine-tune) | —                       | ~44.7 |
| Fine-tune v1                      | 91 (43 + 48 synthetic)  | 37.6  |
| Fine-tune v2                      | 141 (43 + 98 synthetic) | 31.0  |

We also inspect retrieval confidence. Two test questions stayed below 0.4; both were reviewed by hand and judged to have a reasonable match given the fixed 24-document corpus. This repository’s scripts are the same retrieval + QLoRA pipeline used for the Kaggle submission.

## Reproduction

```
C10-team-tana/
├── README.md
├── requirements.txt
├── data/                 # factsheets and Q–A CSVs
├── docs/  and  doc/      # four Cohort Challenge PDFs
└── scripts/              # retrieval_step.py then finetune_clinical_slm.py
```

Run in a Kaggle Notebook with GPU (Settings → Accelerator → T4 x2):

```bash
pip install -q -U -r requirements.txt
```

1. Upload `data/combined_documents.csv`, `data/combined_train_qa.csv`, and the competition `test_questions.csv` as a Kaggle Dataset (this repo uses `devpinheiro/slm-v3`).
2. Run `scripts/retrieval_step.py` — set `DOCS_PATH` and `TEST_PATH`. Writes `test_questions_with_context.csv`.
3. Run `scripts/finetune_clinical_slm.py` in the same session. Trains the LoRA adapter, generates answers, and writes `submission.csv`.

## Appendix

**Team Tana**

- Samuel Pinheiro (Team Lead)
- Onyinyechi Okereke
- **Mentor:** [Add mentor name]

  **Programme:** TRI AI Saturdays, Cohort 10 — Domain-Specific SLM for Medical Advice.

  **Cohort Challenges** (same four PDFs in `docs/` and `doc/`)

- Challenge 1 — `[problem_statement.pdf](docs/problem_statement.pdf)`
- Challenge 2 — `[data_card.pdf](docs/data_card.pdf)`
- Challenge 3 — `[impact_statement_card.pdf](docs/impact_statement_card.pdf)`
- Challenge 4 — `[stakeholder_engagement.pdf](docs/stakeholder_engagement.pdf)`

## References

Full citations for real-sourced documents are in `data/real_documents_all.csv` (`source_url`) and `[docs/data_card.pdf](docs/data_card.pdf)`.

- Gemma 2: [https://huggingface.co/google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)
- Sentence-Transformers MiniLM: [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- WHO and CDC source URLs listed per row in `data/real_documents_all.csv`
- TRI AI Cohort 10 projects: [https://aisaturdayslagos.github.io/cohort_structure/cohort10/projects.html](https://aisaturdayslagos.github.io/cohort_structure/cohort10/projects.html)
