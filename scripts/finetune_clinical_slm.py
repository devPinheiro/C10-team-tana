# ============================================================
# Fine-tune a small language model for clinical Q&A
# Team Tana — TRI AI Saturdays Cohort 10
#
# Run this in a Kaggle Notebook with GPU enabled:
# Settings > Accelerator > GPU T4 x2 (or P100)
# ============================================================

# --- 0. Install dependencies (Kaggle usually has most of these) ---
# !pip install -q -U transformers peft trl bitsandbytes accelerate datasets

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import transformers
print(transformers.__version__)  # should print 4.46.3 exactly
# ============================================================
# 1. LOAD & JOIN DATA
# ============================================================


DOCS_PATH = "/kaggle/input/datasets/devpinheiro/slm-v3/combined_documents.csv"          
TRAIN_QA_PATH = "/kaggle/input/datasets/devpinheiro/slm-v3/combined_train_qa.csv"  
TEST_WITH_CONTEXT_PATH = "/kaggle/working/test_questions_with_context.csv"

documents = pd.read_csv(DOCS_PATH)
train_qa = pd.read_csv(TRAIN_QA_PATH)
test_questions = pd.read_csv(TEST_WITH_CONTEXT_PATH)

train_joined = train_qa.merge(
    documents[["document_id", "text"]], on="document_id", how="left"
).rename(columns={"text": "context"})

assert train_joined["context"].isnull().sum() == 0, "Unjoined rows found — check document_id matches"

# ============================================================
# 2. PROMPT TEMPLATE
# Must match exactly between training and inference
# ============================================================
INSTRUCTION = (
    "Answer the patient question in 1-3 sentences using the context below. "
    "Include a clear action or triage cue if urgent."
)

def build_prompt(context, question):
    return (
        f"Instruction: {INSTRUCTION}\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

train_joined["prompt"] = train_joined.apply(
    lambda r: build_prompt(r["context"], r["question"]), axis=1
)
# Full text the model trains on: prompt + answer + eos
train_joined["full_text"] = train_joined["prompt"] + " " + train_joined["reference_answer"]

train_dataset = Dataset.from_pandas(
    train_joined[["full_text"]].rename(columns={"full_text": "text"})
)

# ============================================================
# 3. MODEL & TOKENIZER
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# Alternatives:
# MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
# MODEL_NAME = "google/gemma-2-2b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# ============================================================
# 4. LoRA CONFIG
# Small rank is enough for a stylistic/format fine-tune on 201 dataset
# ============================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# 5. TRAINING ARGS
# ============================================================
training_args = SFTConfig(
    output_dir="./clinical_slm_lora",
    num_train_epochs=6,             # small data -> more passes, but monitor loss
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=5,
    logging_steps=5,
    save_strategy="epoch",
    bf16=True,             
    report_to="none",
    max_seq_length=512,
    gradient_checkpointing=False, 
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)

trainer.train()

# Save the LoRA adapter (small — a few MB, easy to save as a Kaggle Dataset for reuse)
trainer.save_model("./clinical_slm_lora_final")
tokenizer.save_pretrained("./clinical_slm_lora_final")

# ============================================================
# 6. INFERENCE ON TEST SET
# ============================================================
test_joined = test_questions.copy()
assert "context" in test_joined.columns, (
    "No 'context' column found — run retrieval_step.py first to produce "
    "test_questions_with_context.csv, then point TEST_WITH_CONTEXT_PATH at it."
)

low_conf = test_joined[test_joined.get("retrieval_score", 1.0) < 0.4]
if len(low_conf) > 0:
    print(f"NOTE: {len(low_conf)} test question(s) had low retrieval confidence (<0.4).")
    print("Their generated answers are more likely to be poorly grounded — review manually.")

model.eval()

def generate_answer(context, question, max_new_tokens=60):
    prompt = build_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    full = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the generated answer, after "Answer:"
    answer = full.split("Answer:")[-1].strip()
    return answer

predictions = []
for _, row in test_joined.iterrows():
    ans = generate_answer(row["context"], row["question"])
    predictions.append({"QuestionId": row["QuestionId"], "Answer": ans})

submission = pd.DataFrame(predictions)

# ============================================================
# 7. POST-PROCESS FOR STYLE / LENGTH protects Levenshtein score
# ============================================================
HEDGE_PHRASES = [
    "it's important to note that", "it is important to note that",
    "you may want to consider", "you should consider",
    "please note that", "it should be noted that",
]

def clean_answer(text):
    t = text.strip()
    low = t.lower()
    for phrase in HEDGE_PHRASES:
        if low.startswith(phrase):
            t = t[len(phrase):].strip(" ,.-")
    # Trim to first 3 sentences max
    sentences = t.split(". ")
    if len(sentences) > 3:
        t = ". ".join(sentences[:3])
        if not t.endswith("."):
            t += "."
    return t

submission["Answer"] = submission["Answer"].apply(clean_answer)

# ============================================================
# 8. SAVE SUBMISSION
# ============================================================
submission.to_csv("submission.csv", index=False)
print(submission)