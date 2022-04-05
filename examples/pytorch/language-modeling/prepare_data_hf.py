from datasets import load_dataset, concatenate_datasets, DatasetDict
from datasets.utils.file_utils import DownloadConfig

# langs = ["et", "lv", "lt", "uk", "pl", "fr", "en", "bg"] 

# langs = ["et", "fr", "bg", "en"] # 25000000
# lines_per_lang = 25000000 # to factor out data inbalance issue 
# valid_fraq = 0.0001

langs = ["et", "fr", "lv"] # 25000000
lines_per_lang = 25000
valid_fraq = 00.1


cache_dir = "/gpfs/space/projects/nlpgroup/hf_cache"
savedir = "/gpfs/space/home/maksym95/third-paper/saved_datasets"

raw_datasets = []
for lang in langs:
    d = load_dataset("cc100", 
                    lang=lang,
                    split=f'train[:{lines_per_lang}]',
                    download_config=DownloadConfig(num_proc=100, resume_download=True), 
                    cache_dir=cache_dir)
    d = d.train_test_split(test_size=valid_fraq, shuffle=True)
    d["validation"] = d.pop("test")
    raw_datasets.append(d)

res = {}
res['train'] = concatenate_datasets([d['train'] for d in raw_datasets], split="train").shuffle(seed=42)
res['validation'] = concatenate_datasets([d['validation'] for d in raw_datasets]).shuffle(seed=42)
raw_datasets = DatasetDict(res)

savepath = f"{savedir}/shuf-{'_'.join(langs)}-{len(langs) * lines_per_lang}" 
raw_datasets.save_to_disk(savepath)