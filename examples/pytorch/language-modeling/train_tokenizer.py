from transformers import AutoTokenizer
from datasets import load_from_disk


projdir = "/gpfs/space/home/maksym95/third-paper"
dsetname = "shuf-et_fr_lv-75000"
dataset_path = f"{projdir}/saved_datasets/{dsetname}"
modelname = "xlm-roberta-base"

raw_datasets = load_from_disk(dataset_path)

old_tokenizer = AutoTokenizer.from_pretrained(modelname)

tokenizer = old_tokenizer.train_new_from_iterator(raw_datasets['train'], 52000)
tokenizer.save_pretrained(f"{projdir}/saved_tokenizers/{modelname}-{dsetname}")


