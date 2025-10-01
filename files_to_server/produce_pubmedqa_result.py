# program here produce the result of prediction on pubmedqa dataset after using different finetune parameter
import os, re, argparse
from typing import Dict, Any, List
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import importlib.util
from dotenv import load_dotenv

load_dotenv()

PROMPTS_PY    = "utils/prompts.py"
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", None))
GEN_TEMPERATURE    = float(os.getenv("GEN_TEMPERATURE", None))
GEN_TOP_P          = float(os.getenv("GEN_TOP_P", None))
GEN_DO_SAMPLE      = False if os.environ.get("GEN_DO_SAMPLE", None) == "False" else True
MODEL_DIR          = os.environ.get("MODEL_DIR", None)
OUT_CSV            = os.environ.get("OUT_CSV", None)
BF16               = os.environ.get("BF16", "1") == "1"
TESTING_BENCHMARK  = True if os.environ.get("TESTING_BENCHMARK", None) == "True" else False
# laod parameter
dataset="pubmedqa"
TEST_CSV  = f"Trainset/{dataset}/test.csv" 

TEMPLATE_FILE = f"prompt/{dataset}.txt"
with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
    TEMPLATE_STR = f.read()

spec = importlib.util.spec_from_file_location("prompts", PROMPTS_PY)
prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompts)
pm = prompts.prompts_maker()
test_df  = pd.read_csv(TEST_CSV) 


# # Build prompts for test rows
# def build_prompt_only(row: Dict[str, Any]) -> str:
#     # if label column isn't present, we don't require it here
#     q = row.get("question", "")
#     c = row.get("context", "")
#     return pm.PubMedQA(TEMPLATE_STR, {"question": q, "context": c})

# # Prepare prompts
# need_cols = ["question", "context"]
# for c in need_cols:
#     if c not in test_df.columns:
#         raise ValueError(f"TEST CSV must contain columns {need_cols}, missing: {c}")

# test_prompts = test_df.apply(build_prompt_only, axis=1)


# set the trainer
# trainer
# Generate predictions
def extract_after_answer(text: str) -> str:
    m = re.search(r'Answer:\s*(.*)$', text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    tail = m.group(1).strip()
    first = re.split(r'[\s\.,;:!\?\)\]]+', tail)[0]
    low = first.lower()
    if low.startswith("yes"):   return "Yes"
    if low.startswith("no"):    return "No"
    if low.startswith("maybe"): return "Maybe"
    return first

# check test_df content
for c in ["question","context"]:
     if c not in test_df.columns:
         raise ValueError(f"TEST CSV must contain columns ['question','context'], missing: {c}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dtype = torch.bfloat16 if BF16 and torch.cuda.is_available() else torch.float16 if torch.cuda.is_available() else None
if TESTING_BENCHMARK == True:
    MODEL_DIR = "meta-llama/Llama-3.2-1B-Instruct"
    OUT_CSV = "result_ORG_BENCHMARK.csv"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype if dtype else None)
model.eval()

def build_prompt_only(row: Dict[str, Any]) -> str:
    return pm.pubmedqa(TEMPLATE_STR, {"question": row.get("question",""), "context": row.get("context","")})
prompts_series = test_df.apply(build_prompt_only, axis=1)

preds: List[str] = []
raws:  List[str] = []
@torch.no_grad()
def generate_one(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    ctx = torch.autocast("cuda", dtype=dtype) if dtype and torch.cuda.is_available() else nullcontext()
    with ctx:
        out = model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            do_sample=GEN_DO_SAMPLE,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

from contextlib import nullcontext
for p in prompts_series.tolist():
    g = generate_one(p)
    raws.append(g)
    preds.append(extract_after_answer(g))


os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
out_df = test_df.copy()
out_df["prompt"] = prompts_series
out_df["prediction"] = preds
out_df["raw_generation"] = raws
out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"[INFO] Test predictions saved to: {OUT_CSV}")