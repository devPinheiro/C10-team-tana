# C10 Team Tana: Domain-Specific SLM for Medical Advice

TRI AI Saturdays, Cohort 10. A retrieval-then-generate small language model that answers patient-facing clinical guidance questions in 1–3 evidence-grounded sentences, built for low-resource and frontline health settings.

**Educational benchmark only — not for clinical use.**

**Track:** Domain-Specific SLM for Medical Advice (Kaggle).

**Repo:** [github.com/devPinheiro/C10-team-tana](https://github.com/devPinheiro/C10-team-tana)

## What this repo contains

```
C10-team-tana/
├── README.md
├── requirements.txt
├── data/
│   ├── documents.csv              # 24 synthetic public-health factsheets
│   └── augmented_train_qa_v2.csv  # 141 grounded question–answer pairs
├── docs/
│   ├── Value-led Problem Statement - Domain-Specific Small Language Model for Medical Advice (Team Tana).pdf
│   ├── Team Tana Data Card v2.pdf
│   ├── Impact Statement -  C10 Team Tana.pdf
│   └── Stakeholder Engagement Plan - C10 Team Tana.pdf
└── scripts/
    ├── retrieval_step.py          # attach a factsheet to each test question
    └── finetune_clinical_slm.py   # QLoRA train + write submission.csv
```

Training checkpoints (`clinical_slm_lora/`, `clinical_slm_lora_final/`), `submission.csv`, and `test_questions_with_context.csv` are generated at runtime and are gitignored.

## Cohort documents

| File | Challenge |
| --- | --- |
| [Value-led Problem Statement](docs/Value-led%20Problem%20Statement%20-%20Domain-Specific%20Small%20Language%20Model%20for%20Medical%20Advice%20(Team%20Tana).pdf) | Challenge 1 — problem framing |
| [Team Tana Data Card v2](docs/Team%20Tana%20Data%20Card%20v2.pdf) | Challenge 2 — dataset documentation |
| [Impact Statement](docs/Impact%20Statement%20-%20%20C10%20Team%20Tana.pdf) | Challenge 3 — impact |
| [Stakeholder Engagement Plan](docs/Stakeholder%20Engagement%20Plan%20-%20C10%20Team%20Tana.pdf) | Challenge 4 — stakeholders |

## Dataset

This repository ships the competition-scale corpus plus Team Tana’s v2 augmentation, documented in [Team Tana Data Card v2](docs/Team%20Tana%20Data%20Card%20v2.pdf). All factsheet text is synthetic educational content (CC0-1.0). There are no real patient records.

| File | Rows | Role |
| --- | --- | --- |
| `data/documents.csv` | 24 | Factsheets (~37 words each) with `document_id`, `title`, `topic`, `care_setting`, `population`, `text` |
| `data/augmented_train_qa_v2.csv` | 141 | Questions joined to a gold `document_id` and a short `reference_answer` |

Eight topics, three factsheets each: chronic disease, infectious disease, maternal health, medication safety, nutrition, vaccination, emergency triage, mental-health basics. Care settings include primary care, community, home, hospital, and school. Populations include adult, child, infant, pregnant, and general.

Gold training joins each question to its factsheet by `document_id`. Every one of the 24 documents has at least four training pairs.

The Kaggle notebook currently reads `combined_documents.csv` and `combined_train_qa.csv` from dataset [`devpinheiro/slm-v3`](https://www.kaggle.com/datasets/devpinheiro/slm-v3). Point those paths at the files in `data/` if you are running from this repo instead.

## Pipeline

**1. Retrieval (test only).** Hidden test questions have no gold document. `scripts/retrieval_step.py` filters factsheets by `topic` / `care_setting` / `population`, then ranks remaining texts with [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) cosine similarity. A four-stage fallback (exact metadata → topic+population → topic → full corpus) guarantees every question gets a context. Scores below 0.4 are printed for manual review. Output: `test_questions_with_context.csv`.

**2. Fine-tune.** `scripts/finetune_clinical_slm.py` trains [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) with QLoRA (4-bit NF4, LoRA rank 16 on `q/k/v/o_proj`). Commented alternatives: `microsoft/Phi-3-mini-4k-instruct`, `google/gemma-2-2b-it`.

| Setting | Value |
| --- | --- |
| Epochs | 6 |
| Learning rate | 2e-4, cosine, 5 warmup steps |
| Batch | 2 × 4 gradient accumulation |
| Max sequence length | 512 |
| Gradient checkpointing | off |
| Generation | greedy (`do_sample=False`), max 60 new tokens |

The prompt is fixed between train and inference:

```
Instruction: Answer the patient question in 1-3 sentences using the context below. Include a clear action or triage cue if urgent.
Context: <retrieved factsheet>
Question: <question>
Answer:
```

Post-processing strips hedge openers and caps answers at three sentences before writing `submission.csv`.

QLoRA is used instead of full fine-tuning to limit overfitting on a small pair set. Retrieval-then-generate is used instead of pure generation so every answer stays traceable to a source document.

## Environment

Pin Transformers to **4.46.3**. Transformers 5.x breaks `AutoTokenizer` in this stack (`ModuleNotFoundError: transformers.models.audioflamingo3`) and is incompatible with `trl==0.12.1`.

```bash
pip install -q -r requirements.txt
```

```
transformers==4.46.3
trl==0.12.1
peft==0.13.2
accelerate==1.0.1
bitsandbytes>=0.46.1
datasets==3.1.0
sentence-transformers
```

After installing, restart the notebook kernel and confirm:

```python
import transformers
print(transformers.__version__)  # 4.46.3
```

## Reproduction (Kaggle GPU)

Settings → Accelerator → GPU T4 x2 (or P100).

1. Upload this repo’s `data/` files (or attach dataset `devpinheiro/slm-v3`) plus the competition `test_questions.csv`.
2. In `scripts/retrieval_step.py`, set `DOCS_PATH` and `TEST_PATH`. Run it. Writes `test_questions_with_context.csv`.
3. In `scripts/finetune_clinical_slm.py`, set `DOCS_PATH`, `TRAIN_QA_PATH`, and `TEST_WITH_CONTEXT_PATH` to those files. Run it in the same session. Trains the LoRA adapter, generates answers, and writes `submission.csv`.

## Evaluation

The competition metric is mean Levenshtein distance against 11 hidden-reference test questions (lower is better). Earlier iterations on this pipeline:

| Iteration | Training pairs | Score |
| --- | --- | --- |
| Retrieval baseline (no fine-tune) | — | ~44.7 |
| Fine-tune v1 | 91 | 37.6 |
| Fine-tune v2 | 141 (`augmented_train_qa_v2.csv`) | 31.0 |

The current default base model is Qwen2.5-1.5B-Instruct. Re-run the Kaggle notebook to score a new submission.

## Team

**Team Tana** — TRI AI Saturdays, Cohort 10

- Samuel Pinheiro (Team Lead)
- Onyinyechi Okereke

## References

- Team Tana Data Card v2: [docs/Team Tana Data Card v2.pdf](docs/Team%20Tana%20Data%20Card%20v2.pdf)
- Qwen2.5-1.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- Sentence-Transformers MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- TRI AI Cohort 10 projects: https://aisaturdayslagos.github.io/cohort_structure/cohort10/projects.html
