from datasets import load_dataset, Features, Value

features = Features(
    {
        "conversations": [
            {
                "role": Value("string"),
                "content": Value("string"),
                "reasoning_content": Value("string"),
                "tools": Value("string"),
                "tool_calls": Value("string"),
            }
        ]
    }
)
samples = load_dataset(
    "json",
    data_files="minimind\dataset\sft_t2t_mini.jsonl",
    split="train",
    features=features,
)
print(samples[3])
