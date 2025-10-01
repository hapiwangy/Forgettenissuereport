# -*- coding: utf-8 -*-
"""
Data CSVs are expected in datasets/dreaddit_shuffled_fold0/:
  - shuffled_fold0_train.csv
  - shuffled_fold0_val.csv
  - shuffled_fold0_test.csv
"""
import os
import re
import pandas as pd
from typing import Dict, Any, List

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import argparse

# -------------------- Args & Config --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save continued finetune outputs")
parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint path to resume from")
parser.add_argument("--epochs", type=float, default=None, help="Num train epochs for this run (overrides EPOCHS env; default 1)")
args = parser.parse_args()

MODEL_NAME = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
BF16 = os.environ.get("BF16", "1") == "1"
USE_4BIT = os.environ.get("USE_4BIT", "1") == "1"

OUTPUT_DIR = args.output_dir or os.environ.get("OUTPUT_DIR", "runs/outputs-trl-llama32-1b-dreaddit")
# Default to previous PubMedQA checkpoint unless overridden
RESUME_FROM = args.resume_from or os.environ.get("RESUME_FROM", r"runs/outputs-trl-llama32-1b-pubmedqa/checkpoint-123")

TRAIN_CSV = r"datasets/dreaddit_shuffled_fold0/shuffled_fold0_train.csv"
VAL_CSV   = r"datasets/dreaddit_shuffled_fold0/shuffled_fold0_val.csv"
TEST_CSV  = r"datasets/dreaddit_shuffled_fold0/shuffled_fold0_test.csv"

TEMPLATE_FILE = "prompt/dreaddit.txt"
PROMPTS_PY    = "utils/prompts.py"

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "1024"))
PER_DEVICE_BATCH = int(os.environ.get("BATCH", "8"))
GRAD_ACCUM = int(os.environ.get("ACCUM", "2"))
LR = float(os.environ.get("LR", "2e-4"))
EPOCHS = args.epochs if args.epochs is not None else float(os.environ.get("EPOCHS", "3"))
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

# -------------------- Import prompts_maker --------------------
import importlib.util
spec = importlib.util.spec_from_file_location("prompts", PROMPTS_PY)
prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompts)  # exposes prompts.prompts_maker()

# -------------------- Load template --------------------
with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
    TEMPLATE_STR = f.read()

# -------------------- Data helpers --------------------
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["subreddit", "text", "label"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV {path} must contain columns: {required}")
    return df

# DREDDIT: label 1 => Stress, 0 => No Stress
MAP_DREADDIT = {
    "1": "Stress", "0": "No Stress",
    1: "Stress", 0: "No Stress",
    "A": "Stress", "B": "No Stress",
    "a": "Stress", "b": "No Stress",
    "stress": "Stress", "no stress": "No Stress",
    "Stress": "Stress", "No Stress": "No Stress",
}

def normalize_label(x: Any) -> str:
    x = str(x).strip()
    return MAP_DREADDIT.get(x, x)

pm = prompts.prompts_maker()

def build_example(row: Dict[str, Any]) -> Dict[str, str]:
    input_text = pm.dreaddit(TEMPLATE_STR, {"subreddit": row["subreddit"], "text": row["text"]})
    target = normalize_label(row["label"])
    full_text = f"{input_text} {target}".strip()
    return {"text": full_text}

# -------------------- Prepare datasets --------------------
train_df = load_df(TRAIN_CSV)
val_df   = load_df(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

train_out = train_df.apply(build_example, axis=1, result_type="expand")
val_out   = val_df.apply(build_example, axis=1, result_type="expand")

ds = DatasetDict({
    "train": Dataset.from_pandas(train_out, preserve_index=False),
    "validation": Dataset.from_pandas(val_out, preserve_index=False)
})

# -------------------- Tokenizer & Model --------------------
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=(torch.bfloat16 if BF16 else torch.float16),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
else:
    bnb_config = None

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# -------------------- LoRA (QLoRA) --------------------
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
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
    save_total_limit=3,
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
def formatting_func(example):
    return example["text"]   # 回傳字串，而不是 list

# -------------------- Trainer --------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    args=sft_config,
    peft_config=peft_config,
    processing_class=tokenizer,
    formatting_func=formatting_func
)

# -------------------- Resume Train --------------------
if not os.path.isdir(RESUME_FROM):
    raise FileNotFoundError(f"RESUME_FROM checkpoint folder not found: {RESUME_FROM}")

print(f"[INFO] Resuming from: {RESUME_FROM}")
trainer.train(resume_from_checkpoint=RESUME_FROM)

# -------------------- Save adapter/tokenizer --------------------
adapter_dir = os.path.join(OUTPUT_DIR, "adapter")
os.makedirs(adapter_dir, exist_ok=True)
trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[INFO] Continued training done. Adapter saved to: {adapter_dir}")

# -------------------- Prediction on TEST split --------------------
def build_prompt_only(row: Dict[str, Any]) -> str:
    return pm.dreaddit(TEMPLATE_STR, {"subreddit": row.get("subreddit", ""), "text": row.get("text", "")})

need_cols = ["subreddit", "text"]
for c in need_cols:
    if c not in test_df.columns:
        raise ValueError(f"TEST CSV must contain columns {need_cols}, missing: {c}")

test_prompts = test_df.apply(build_prompt_only, axis=1)

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
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_after_answer(text: str) -> str:
    m = re.search(r'Answer:\s*(.*)$', text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    tail = m.group(1).strip()
    first_line = tail.splitlines()[0].strip()
    normalized = normalize_label(first_line)
    if normalized in ("Stress", "No Stress"):
        return normalized
    first_token = re.split(r'[\s\.,;:!\?\)\]]+', first_line)[0]
    candidate = normalize_label(first_token)
    if candidate in ("Stress", "No Stress"):
        return candidate
    return first_line

for prompt_text in test_prompts.tolist():
    generation = generate_answer(prompt_text)
    raw_generations.append(generation)
    pred_answers.append(extract_after_answer(generation))

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "second_fine_tuneresults.csv")
out_df = test_df.copy()
out_df["prompt"] = test_prompts
out_df["prediction"] = pred_answers
out_df["raw_generation"] = raw_generations
out_df.to_csv(out_path, index=False, encoding="utf-8")
print(f"[INFO] Test predictions saved to: {out_path}")
