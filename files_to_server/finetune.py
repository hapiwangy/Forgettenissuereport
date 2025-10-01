# import the package
import torch
import os
from typing import Dict, Any, List
from utils import prompt_maker
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

load_dotenv()

# dataset2model
## could be changed if datasets are being added or the order of the finetune changes
data2model = {
    'pubmedqa':"meta-llama/Llama-3.2-1B-Instruct",
    'dreaddit':"runs/outputs-trl-llama32-1b-pubmedqa"
}

# prompts related
PROMPTS_PY    = "utils/prompts.py"


if __name__ == "__main__":

    # dataset setting
    dataset = os.environ.get("dataset", None)
    TRAIN_CSV = f"Trainset/{dataset}/train.csv"
    VAL_CSV   = f"Trainset/{dataset}/val.csv"   


    # case > since dataset may contain so many different data set, we should just pre-process the data first together based on their dataset.
    # TEMPLATE_FILE = f"prompt/{dataset}.txt"
    # if there are new datasets, 
    potential_datasset = ['dreaddit', 'pubmedqa']
    test = prompt_maker.promptMaker(potential_datasset)
    ds = test.processingdataset(TRAIN_CSV,VAL_CSV)




    # prompt setting
    BF16 = os.environ.get("BF16", "1") == "1"
    # result output
    OUTPUT_DIR = f"runs/outputs-trl-llama32-1b-{dataset}"
    MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", None))
    PER_DEVICE_BATCH = int(os.environ.get("BATCH", None))
    GRAD_ACCUM = int(os.environ.get("ACCUM", None))
    LR = float(os.environ.get("LR", None))
    EPOCHS = int(os.environ.get("EPOCHS", None))
    SAVE_STEPS = int(os.environ.get("SAVE_STEPS", None))

    # choosing strategy 
    GEN_MAX_NEW_TOKENS = int(os.environ.get("GEN_MAX_NEW_TOKENS", None))
    GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", None))
    GEN_TOP_P = float(os.environ.get("GEN_TOP_P", None))
    GEN_DO_SAMPLE = False if os.environ.get("GEN_DO_SAMPLE", None) == "False" else True

    # make directory if certain not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # set the model
    ## beaware of the naming of the dataset should follow the format of orgdataset-"changes"
    MODEL_NAME = data2model[dataset.split('-')[0]]
    print(f"current use model: {MODEL_NAME} as the base model to finetuning")
    # model and tokenized
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
    # traininf config
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
    def formatting_fun(example):
    # SFTTrainer need return list[str]
        return example["text"]

    # trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=sft_config,
        processing_class=tokenizer,     # 取代舊的 tokenizer=
        formatting_func=formatting_fun # 取代舊的 dataset_text_field=
    )
    # do the training
    trainer.train()

    # save model 
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] Training done. Model saved to: {OUTPUT_DIR}")

    

