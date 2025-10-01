# -*- coding: utf-8 -*-
"""
Data CSVs are expected in datasets/PubMedQa_cleaning/:
  - pqa_labeled_train.csv
  - pqa_labeled_val.csv
  - pqa_labeled_test.csv
"""
import torch
import os
import re
import math
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Dict, Any, List

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# -------------------- Config --------------------
MODEL_NAME = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
BF16 = os.environ.get("BF16", "1") == "1"

TRAIN_CSV = r"datasets/PubMedQa_cleaning/pqa_labeled_train.csv"
VAL_CSV   = r"datasets/PubMedQa_cleaning/pqa_labeled_val.csv"     # kept for completeness; not used for eval
TEST_CSV  = r"datasets/PubMedQa_cleaning/pqa_labeled_test.csv"    # NEW: will be used for prediction after training

TEMPLATE_FILE = "prompt/PubMedQA.txt"
PROMPTS_PY    = "utils/prompts.py"

OUTPUT_DIR = "runs/outputs-trl-llama32-1b-pubmedqa"
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1024"))
PER_DEVICE_BATCH = int(os.environ.get("BATCH", "8"))
GRAD_ACCUM = int(os.environ.get("ACCUM", "2"))
LR = float(os.environ.get("LR", "2e-4"))
EPOCHS = float(os.environ.get("EPOCHS", "3"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "200"))

GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", "8"))
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.0"))
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", "1.0"))
GEN_DO_SAMPLE = os.environ.get("GEN_DO_SAMPLE", "0") == "1"

# -------------------- Ensure directories exist --------------------
for path in (
    os.path.dirname(TRAIN_CSV),
    os.path.dirname(VAL_CSV),
    os.path.dirname(TEST_CSV),
    os.path.dirname(TEMPLATE_FILE),
    os.path.dirname(PROMPTS_PY),
    OUTPUT_DIR,
):
    if path:
        os.makedirs(path, exist_ok=True)

# -------------------- Import your prompts_maker --------------------
import importlib.util
spec = importlib.util.spec_from_file_location("prompts", PROMPTS_PY)
prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompts)  # exposes prompts.prompts_maker()

# -------------------- Load template --------------------
with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
    TEMPLATE_STR = f.read()

# -------------------- Data loading helpers --------------------
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["question", "context", "final_decision"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV {path} must contain columns: {required}")
    return df

MAP_ABC = {
    "A": "Yes", "B": "No", "C": "Maybe",
    "a": "Yes", "b": "No", "c": "Maybe",
    "yes": "Yes", "no": "No", "maybe": "Maybe",
    "Yes": "Yes", "No": "No", "Maybe": "Maybe"
}

def normalize_label(x: str) -> str:
    x = str(x).strip()
    return MAP_ABC.get(x, x)

# -------------------- Build SFT text --------------------
pm = prompts.prompts_maker()

def build_example(row: Dict[str, Any]) -> Dict[str, str]:
    # Build input prompt with "Answer:" suffix
    input_text = pm.PubMedQA(TEMPLATE_STR, {"question": row["question"], "context": row["context"]})
    target = normalize_label(row["final_decision"])
    full_text = f"{input_text} {target}".strip()
    return {"text": full_text}

# -------------------- Prepare datasets --------------------
train_df = load_df(TRAIN_CSV)
val_df   = load_df(VAL_CSV)         # not used for eval but kept for completeness
test_df  = pd.read_csv(TEST_CSV)    # test may or may not contain label; we won't require it for prediction

train_out = train_df.apply(build_example, axis=1, result_type="expand")
val_out   = val_df.apply(build_example, axis=1, result_type="expand")

ds = DatasetDict({
    "train": Dataset.from_pandas(train_out, preserve_index=False),
    "validation": Dataset.from_pandas(val_out, preserve_index=False)
})

# -------------------- Tokenizer & Model --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model_kwargs: Dict[str, Any] = {}
if torch.cuda.is_available():
    if BF16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    **model_kwargs
)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        max_length=MAX_SEQ_LEN,
        truncation=True
    )

# -------------------- Training Config --------------------
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=20,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    bf16=BF16,
    fp16=not BF16,
    do_eval=True,               
    eval_strategy="steps",      
    eval_steps=200,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
    packing=False
)

# -------------------- Define formatting function --------------------
def formatting_fun(example):
    # SFTTrainer 需要回傳一個 list[str]
    return example["text"]

# -------------------- Trainer --------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    args=sft_config,
    processing_class=tokenizer,     # 取代舊的 tokenizer=
    formatting_func=formatting_fun # 取代舊的 dataset_text_field=
)
# -------------------- Train --------------------
trainer.train()

# -------------------- Save model/tokenizer --------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[INFO] Training done. Model saved to: {OUTPUT_DIR}")

# -------------------- Prediction on TEST split --------------------
# Build prompts for test rows
def build_prompt_only(row: Dict[str, Any]) -> str:
    # if label column isn't present, we don't require it here
    q = row.get("question", "")
    c = row.get("context", "")
    return pm.PubMedQA(TEMPLATE_STR, {"question": q, "context": c})

# Prepare prompts
need_cols = ["question", "context"]
for c in need_cols:
    if c not in test_df.columns:
        raise ValueError(f"TEST CSV must contain columns {need_cols}, missing: {c}")

test_prompts = test_df.apply(build_prompt_only, axis=1)

# Generate predictions
trainer.model.eval()

pred_answers: List[str] = []
raw_generations: List[str] = []

@torch.no_grad()
def generate_answer(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
    dtype = torch.bfloat16 if BF16 else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            do_sample=GEN_DO_SAMPLE,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def extract_after_answer(text: str) -> str:
    # Extract text after the last "Answer:" occurrence, then clean to (Yes|No|Maybe) if possible
    m = re.search(r'Answer:\s*(.*)$', text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    tail = m.group(1).strip()
    # take first token-like word
    first = re.split(r'[\s\.,;:!\?\)\]]+', tail)[0]
    # normalize
    low = first.lower()
    if low.startswith("yes"):
        return "Yes"
    if low.startswith("no"):
        return "No"
    if low.startswith("maybe"):
        return "Maybe"
    # fallback to raw
    return first

for p in test_prompts.tolist():
    gen = generate_answer(p)
    raw_generations.append(gen)
    pred_answers.append(extract_after_answer(gen))

# Save predictions
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "first_finetune_results.csv")
out_df = test_df.copy()
out_df["prompt"] = test_prompts
out_df["prediction"] = pred_answers
out_df["raw_generation"] = raw_generations
out_df.to_csv(out_path, index=False, encoding="utf-8")
print(f"[INFO] Test predictions saved to: {out_path}")







