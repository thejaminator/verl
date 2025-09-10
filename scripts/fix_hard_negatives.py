# %%

raise ValueError("read details before running")

# note: move this up to the verl dir before running. Backup your data before running, this script may have sharp edges

# %%

from tqdm import tqdm
import json

good_filenames = [
    "data_good/qwen_hard_negatives_0_20000_layer_percent_25.jsonl",
    "data_good/qwen_hard_negatives_0_20000_layer_percent_50.jsonl",
    "data_good/qwen_hard_negatives_0_20000_layer_percent_75.jsonl",
]

for good_filename in good_filenames:
    good_sae_info_filename = good_filename.replace("data_good", "data").replace("20000", "30")

    with open(good_sae_info_filename, "r") as f:
        good_sae_info = [json.loads(line) for line in f]

    sae_info = good_sae_info[0]["sae_info"]

    all_data = []

    max_lines = 30
    max_lines = None

    with open(good_filename, "r") as f:
        for i, line in tqdm(enumerate(f), "loading"):
            if max_lines is not None and i >= max_lines:
                break

            data = json.loads(line)

            data["sae_info"] = sae_info

            if "sae_layer" in data:
                data.pop("sae_layer")

            all_data.append(data)

    with open(good_filename, "w") as f:
        for i, data in tqdm(enumerate(all_data), "saving"):
            if max_lines is not None and i >= max_lines:
                break

            f.write(json.dumps(data) + "\n")

    print(f"Updated {good_filename}")

# %%
# %%

# import json

# good_filenames = [
#     "data_good/qwen_hard_negatives_0_20000_layer_percent_25.jsonl",
#     "data_good/qwen_hard_negatives_0_20000_layer_percent_50.jsonl",
#     "data_good/qwen_hard_negatives_0_20000_layer_percent_75.jsonl",
#     # "data_good/qwen_hard_negatives_0_20000_layer_percent_50_sft_data_gpt-5-mini-2025-08-07.jsonl",
# ]

# idx = 1

# first_filename = good_filenames[idx]
# good_sae_info_filename = good_filenames[idx].replace("data_good", "data").replace("20000", "30")

# max_lines = 30
# data = []
# with open(good_filenames[idx], "r") as f:
#     for i in range(max_lines):
#         line = f.readline()
#         if line == "":
#             break
#         data.append(json.loads(line))

# with open(good_sae_info_filename, "r") as f:
#     good_sae_info = [json.loads(line) for line in f]

# # %%

# print(good_sae_info[0].keys())
# print(good_sae_info[0]["sae_info"])

# print("__")
# print(data[1].keys())
# print(data[1]["sae_id"])
# print(data[1]["sae_info"])

# # %%
