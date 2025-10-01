# Model Training Report

## Experiment Overview
- Base model: `meta-llama/Llama-3.2-1B-Instruct` with 4-bit quantization and LoRA adapters (`r=16`, `lora_alpha=32`, dropout `0.05`).
- Two-stage finetuning pipeline: first on PubMedQA (medical QA, 3-class), then continued training on Dreaddit (stress detection, binary).
- Prompt templates reside in `prompt/PubMedQA.txt` and `prompt/dreaddit.txt` for consistent instruction formatting.

## Stage 1 – PubMedQA Supervised Finetuning
- Script: `first_finetune.py` (`first_finetune.py:24-264`).
- Data: `datasets/PubMedQa_cleaning/pqa_labeled_train/val/test.csv`.
- Key hyperparameters: batch size 8, gradient accumulation 2, learning rate 2e-4, epochs 3, max sequence length 1024, save/eval every 200 steps. Training uses bf16 autocast when available.
- Training output: `result_on_server/runs/outputs-trl-llama32-1b-pubmedqa` (adapter + tokenizer).
- Training summary: final loss 1.0976, mean token accuracy 0.756 (`report_metrics.json:152-160`).
- Test set (105 samples): accuracy 0.6286, macro F1 0.3933; class performance — Yes F1 0.7465, No F1 0.4333, Maybe none captured (`report_metrics.json:3-55`).

## Stage 2 – Dreaddit Continued Finetuning
- Script: `second_finetune.py` (`second_finetune.py:29-257`).
- Initialization: resumes from PubMedQA checkpoint `checkpoint-123`.
- Data: `datasets/dreaddit_shuffled_fold0` train/val/test splits with stress labels.
- Hyperparameters: same as Stage 1 (batch 8, grad accum 2, lr 2e-4, epochs 3, max seq 1024) with identical LoRA setup.
- Training summary: final loss 1.8255, mean token accuracy 0.6131; best eval loss 1.9811 at step 400 (`report_metrics.json:162-177`).
- Test set (715 samples): accuracy 0.8517, macro F1 0.8516; balanced predictions across Stress/No Stress (`report_metrics.json:57-95`).

## Cross-Task Evaluation After Stage 2
- PubMedQA validation set (245 samples) re-run with Dreaddit adapter.
- Accuracy 0.5918, macro F1 0.3126; model predicts Yes for 93% of samples, missing Maybe entirely and recalling No at 12% (`report_metrics.json:96-148`).
- Indicates catastrophic forgetting of PubMedQA after domain shift.

## Supporting Artifacts
- Metrics generator: `generate_report_summary.py` consolidates CSV outputs and trainer logs into `report_metrics.json` (`generate_report_summary.py:10-204`).
- Inference examples: `check result/inference_example.py`, `check result/inference_example_2.py` show how adapters are loaded for PubMedQA and Dreaddit respectively.

## Observations & Next Steps
- Stage 1 performance skewed toward Yes responses; investigate prompt template or label normalization to recover Maybe predictions.
- Severe forgetting post Stage 2; consider rehearsal with PubMedQA data, multi-task finetuning, or regularization (e.g., L2 on LoRA deltas) to retain earlier knowledge.
- Track eval metrics during training (enable `report_to` for logging) and consider stratified sampling to balance PubMedQA labels.
