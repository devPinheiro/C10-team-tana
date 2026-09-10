# ============================================================
# Retrieval step: attach 'context' to test_questions.csv
# Filters documents.csv by topic/care_setting/population metadata,
# then ranks candidates by semantic similarity to the question.
# ============================================================

# !pip install -q -U sentence-transformers

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

DOCS_PATH = "/kaggle/input/<your-dataset>/documents.csv"
TEST_PATH = "/kaggle/input/<your-dataset>/test_questions.csv"

documents = pd.read_csv(DOCS_PATH)
test_questions = pd.read_csv(TEST_PATH)

# ------------------------------------------------------------
# 1. Load embedding model — small, fast, free on Kaggle CPU/GPU
# ------------------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embedder.encode(
    documents["text"].tolist(), convert_to_tensor=True, show_progress_bar=True
)

# ------------------------------------------------------------
# 2. Retrieval function: filter by metadata first, then rank by similarity
# Falls back to widening the filter if nothing matches exactly,
# so no question is ever left without a retrieved context.
# ------------------------------------------------------------
def retrieve_context(question, topic, care_setting, population, top_k=1):
    candidates = documents.copy()
    candidate_embeds = doc_embeddings

    # Stage 1: exact match on topic + care_setting + population
    mask = (
        (candidates["topic"] == topic)
        & (candidates["care_setting"] == care_setting)
        & (candidates["population"] == population)
    )
    if mask.sum() == 0:
        # Stage 2: relax to topic + population match only
        mask = (candidates["topic"] == topic) & (candidates["population"] == population)
    if mask.sum() == 0:
        # Stage 3: relax to topic match only
        mask = candidates["topic"] == topic
    if mask.sum() == 0:
        # Stage 4: no metadata match at all — use full corpus
        mask = pd.Series([True] * len(candidates))

    filtered = candidates[mask]
    filtered_idx = filtered.index.to_numpy()
    filtered_embeds = doc_embeddings[filtered_idx]

    q_embedding = embedder.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_embedding, filtered_embeds)[0]

    top_result_idx = int(scores.argmax())
    best_row = filtered.iloc[top_result_idx]

    return best_row["document_id"], best_row["text"], float(scores[top_result_idx])

# ------------------------------------------------------------
# 3. Apply retrieval to every test question
# ------------------------------------------------------------
retrieved_doc_ids = []
retrieved_contexts = []
retrieved_scores = []

for _, row in test_questions.iterrows():
    doc_id, context, score = retrieve_context(
        row["question"], row["topic"], row["care_setting"], row["population"]
    )
    retrieved_doc_ids.append(doc_id)
    retrieved_contexts.append(context)
    retrieved_scores.append(score)

test_questions["document_id"] = retrieved_doc_ids
test_questions["context"] = retrieved_contexts
test_questions["retrieval_score"] = retrieved_scores

# ------------------------------------------------------------
# 4. Sanity check before moving to inference
# ------------------------------------------------------------
print(test_questions[["QuestionId", "question", "document_id", "retrieval_score"]].to_string())

low_confidence = test_questions[test_questions["retrieval_score"] < 0.4]
if len(low_confidence) > 0:
    print(f"\nWARNING: {len(low_confidence)} question(s) have low retrieval confidence (<0.4).")
    print("Review these manually — the wrong context may have been retrieved:")
    print(low_confidence[["QuestionId", "question", "document_id", "retrieval_score"]].to_string())

test_questions.to_csv("test_questions_with_context.csv", index=False)
