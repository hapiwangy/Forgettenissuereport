# here contains method that cutting with datas
- original data are put in the datasets directory
- train:test:val = 65%:25%:10%
current training order:
- pubmedqa > dreaddit

# explain
## datasets_naming
- orginal dataset: only names
- new dataset: names followed by the '-', so that model decision wouldn't go wrong
## adding new dataset/ changing order
- changes the 'data2model' variable under 'finetune.py' so that can choose the right model to go finetuning