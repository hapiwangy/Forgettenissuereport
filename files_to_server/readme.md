# how to use the files
based on the dataset, change the file in the .env files to do the finetuning
to each dataset
> run python finetune.py to get the runs/outputs-trl-llama32-1b-{datasetname} (which contains the training result)
> run python produce_pubmedqa.py to get the result that based on the finetune from last stage