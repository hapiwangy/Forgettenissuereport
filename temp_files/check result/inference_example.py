import os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_DIR = "result_on_server/runs/outputs-trl-llama32-1b-pubmedqa/adapter"  # 或改為 checkpoint-123
TOKENIZER_DIR = "result_on_server/runs/outputs-trl-llama32-1b-pubmedqa"         # 用你保存好的 tokenizer

USE_4BIT = True
BF16 = os.environ.get("BF16", "1") == "1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    bnb_4bit_compute_dtype=(torch.bfloat16 if BF16 else torch.float16),
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

# 你的 PubMedQA 模板
template = open("prompt/PubMedQA.txt", "r", encoding="utf-8").read()

def build_prompt(question: str, context: str) -> str:
    return template.format(question=question, context=context)

@torch.no_grad()
def generate(prompt: str, max_new_tokens=8, temperature=0.0, top_p=1.0, do_sample=False):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    dtype = torch.bfloat16 if BF16 else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def extract_answer(text: str) -> str:
    # 1) 取最後一個 "Answer:" 後面的內容（避免前面 prompt 或模型重複的影響）
    lower = text.lower()
    idx = lower.rfind("answer:")
    tail = text[idx + len("answer:"):].strip() if idx != -1 else text

    # 2) 去掉前置標點與括號，常見形式如 "(A)", "A.", "A)" 等
    tail = re.sub(r"^[\s\-\(\[\{:\.\)\]]+", "", tail)

    # 3) 先嘗試直接找 Yes/No/Maybe（不分大小寫），取第一個出現者
    m = re.search(r"\b(yes|no|maybe)\b", tail, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()

    # 4) 若沒有，嘗試解析 A/B/C 及其常見變形
    #    支援格式：A / (A) / A. / A) / option A / choice: B 等
    m2 = re.search(r"\b(?:option|choice)?\s*\(?\s*([abc])\s*[\.)\]:;]?\b", tail, flags=re.IGNORECASE)
    if m2:
        letter = m2.group(1).lower()
        return {"a": "Yes", "b": "No", "c": "Maybe"}[letter]

    # 5) 仍無法解析：在整段文本中回退搜尋最後一個 Yes/No/Maybe
    m3 = re.findall(r"\b(yes|no|maybe)\b", text, flags=re.IGNORECASE)
    if m3:
        return m3[-1].capitalize()

    # 6) 再退一步：擷取 tail 的第一個 token 當作回傳（可能是 "A" / "B" / "C" 之外）
    first = re.split(r"[\s\.,;:!\?\)\]]+", tail)[0]
    first = first.strip("()[]{}:.- ")
    if first.lower() in ("a", "b", "c"):
        return {"a": "Yes", "b": "No", "c": "Maybe"}[first.lower()]
    return first or ""

import pandas as pd
df = pd.read_csv(r"D:\finding jobs notes and usc things\USC projects\forgettingissue\temp_files\datasets\PubMedQa_cleaning\pqa_labeled_test.csv")
dicts = {'pred_label':[]}
for x in range(int(df.shape[0])):
    test_data = df.iloc[x]
    q = test_data['question']
    c = test_data['context']
    prompt = build_prompt(q, c)
    gen = generate(prompt)
    dicts['pred_label'].append(extract_answer(gen))
dfs = pd.DataFrame(dicts)
dfs.to_csv(r'first_result.csv')
    
# from peft import PeftModel
# from transformers import AutoModelForCausalLM

# base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
# merged = PeftModel.from_pretrained(base, ADAPTER_DIR)
# merged = merged.merge_and_unload()  # 權重合併
# merged.save_pretrained("runs/merged-llama32-1b-pubmedqa")
# # 之後用 transformers 直接從 runs/merged-llama32-1b-pubmedqa 載入即可
