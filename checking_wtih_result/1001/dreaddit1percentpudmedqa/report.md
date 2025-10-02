# Training Report (Auto-detected)

**Epochs:** 3
**Global steps:** 444
**Train batch size:** 8
**Logging steps:** 20
**Save steps:** 200

## Final Training Snapshot
- step: 440
- loss: 0.3504
- mean_token_accuracy: 0.9054
- entropy: 0.4505

## Final Eval Snapshot
- step: 400
- eval_loss: 3.2562
- eval_mean_token_accuracy: 0.5267
- eval_entropy: 1.1630

## Notes
- 若 `available_columns.txt` 裡沒有任何 `eval_` 欄位，代表你這次訓練沒有做評估（或還沒到評估步），屬正常現象。
- 本腳本會自動略過不存在的欄位，不會報錯。