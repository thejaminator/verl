# %%

raise ValueError("read details before running")

# note: move this up to the verl dir before running. Backup your data before running, this script may have sharp edges

# %%

import json

import pydantic
from pydantic import BaseModel
from tqdm import tqdm


class SAEInfo(BaseModel):
    sae_width: int
    sae_layer: int
    sae_layer_percent: int
    sae_filename: str
    sae_repo_id: str

class SAEExplained(BaseModel):
    sae_id: int
    explanation: str
    positive_examples: list[str]
    negative_examples: list[str]
    f1: float
    sae_info: SAEInfo


layer_percents = [25, 50, 75]


for layer_percent in layer_percents:
    filename = f"data/qwen_hard_negatives_0_20000_layer_percent_{layer_percent}_sft_data_gpt-5-mini-2025-08-07.jsonl"
    good_sae_info_filename = f"data/qwen_hard_negatives_0_30_layer_percent_{layer_percent}.jsonl"

    with open(good_sae_info_filename, "r") as f:
        good_sae_info = [json.loads(line) for line in f]

    sae_info = good_sae_info[0]["sae_info"]

    all_data = []

    with open(filename, "r") as f:
        for line in tqdm(f, "loading"):
            data = json.loads(line)
            data["sae_info"] = sae_info
            all_data.append(data)






good_sae_info_filename

with open(filenames[1], "r") as f:
    data = [json.loads(line) for line in f]

# %%

print(data[0])

# %%
