from datasets import load_dataset

samples = load_dataset(
    "json", data_files="minimind\dataset\pretrain_t2t_mini.jsonl", split="train"
)
print(samples[3])
