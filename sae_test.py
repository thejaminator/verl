# %% 

%load_ext autoreload
%autoreload 2

# %%
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from create_hard_negatives_v2 import *

filename = "data/qwen_hard_negatives_0_30_layer_percent_50.jsonl"

with open(filename, "r") as f:
    data = [json.loads(line) for line in f]

# %%

num_display = 10

idx = 1

print(data[idx]["activations"]["sae_id"])
print(data[idx]["activations"]["sentences"][0])
print(data[idx]["hard_negatives"][0])

for i,sentence in enumerate(data[idx]["activations"]["sentences"]):
    print(f"\nSentence {i}:, ", sentence["max_act"])
    print("".join(sentence["tokens"]))

for i,sentence in enumerate(data[idx]["hard_negatives"][0]["sentences"]):
    print(f"\nSentence {i}:, ", sentence["max_act"])
    print("".join(sentence["tokens"]))

# %%

sae_repo_id = "adamkarvonen/qwen3-8b-saes"
sae_layer_percent = 50
sae_info = get_sae_info(sae_repo_id, sae_layer_percent)

print(sae_info)

device = torch.device("cuda")
dtype = torch.bfloat16

# %%

# %%

model_name = "Qwen/Qwen3-8B"

model, tokenizer, sae, submodule = load_model_and_sae(
    model_name, sae_repo_id, sae_info.sae_filename, sae_info.sae_layer
)
# %%
max_acts_data = load_max_acts_data(
    model_name=model_name,
    sae_layer=sae_info.sae_layer,
    sae_width=sae_info.sae_width,
    layer_percent=sae_info.sae_layer_percent,
    context_length=32,
)
# %%

print(max_acts_data.keys())
# %%
for feature_idx in range(10):
    print(max_acts_data["max_acts"][feature_idx].max())


# %%

first_sentence = tokenizer.decode(max_acts_data["max_tokens"][0, 0])
sentences = [first_sentence]

print(tokenizer.decode(max_acts_data["max_tokens"][0, 0]))
print(max_acts_data["max_acts"][0, 0])

print(max_acts_data["max_tokens"].shape)
print(max_acts_data["max_acts"].shape)
# %%
acts = compute_sae_activations_for_sentences(
    model,
    tokenizer,
    sae,
    submodule,
    sentences,
    target_feature_idx=0,
    batch_size=1,
)
print(acts)

print(first_sentence)
    
# %%
tokens = tokenizer(sentences, return_tensors="pt", add_special_tokens=False).to(device)
print(tokens)
acts_BLD = collect_activations(model, submodule, tokens)
print(acts_BLD)
encoded_acts_BLF = sae.encode(acts_BLD)
print(encoded_acts_BLF[:, :, 0])
# %%
