# Training Report (Auto-detected)

**Epochs:** 3
**Global steps:** 123
**Train batch size:** 8
**Logging steps:** 20
**Save steps:** 200

## Final Training Snapshot
- step: 120
- loss: 0.2378
- mean_token_accuracy: 0.9390
- entropy: 0.3034

## Final Eval Snapshot
- step: N/A
- eval_loss: N/A
- eval_mean_token_accuracy: N/A
- eval_entropy: N/A

## Notes
- 若 `available_columns.txt` 裡沒有任何 `eval_` 欄位，代表你這次訓練沒有做評估（或還沒到評估步），屬正常現象。
- 本腳本會自動略過不存在的欄位，不會報錯。