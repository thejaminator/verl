# %%

import gc

import torch
from circuitsvis.activations import text_neuron_activations
from IPython.display import clear_output, display
from transformers import AutoTokenizer

from create_hard_negatives_v2 import get_sae_info, load_max_acts_data

# %%
model_name = "Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

sae_repo_id = "adamkarvonen/qwen3-8b-saes"
sae_layer_percent = 50
sae_info = get_sae_info(sae_repo_id, sae_layer_percent)


# %%
max_acts_data = load_max_acts_data(model_name, sae_info.sae_layer, sae_info.sae_width, sae_info.sae_layer_percent)

max_tokens_FKL = max_acts_data["max_tokens"]
max_activations_FKL = max_acts_data["max_acts"]
# %%


def _list_decode(x: torch.Tensor):
    assert len(x.shape) == 1 or len(x.shape) == 2
    # Convert to list of lists, even if x is 1D
    if len(x.shape) == 1:
        x = x.unsqueeze(0)  # Make it 2D for consistent handling

    # Convert tensor to list of list of ints
    token_ids = x.tolist()

    # Convert token ids to token strings
    return [tokenizer.batch_decode(seq, skip_special_tokens=False) for seq in token_ids]


def create_html_activations(
    selected_tokens_FKL: list[str],
    selected_activations_FKL: list[torch.Tensor],
    num_display: int = 10,
    k: int = 5,
) -> list:
    all_html_activations = []

    for i in range(num_display):
        selected_activations_KL11 = [selected_activations_FKL[i, k, :, None, None] for k in range(k)]
        selected_tokens_KL = selected_tokens_FKL[i]
        selected_token_strs_KL = _list_decode(selected_tokens_KL)

        for k in range(len(selected_token_strs_KL)):
            if "<s>" in selected_token_strs_KL[k][0] or "<bos>" in selected_token_strs_KL[k][0]:
                selected_token_strs_KL[k][0] = "BOS>"

        html_activations = text_neuron_activations(selected_token_strs_KL, selected_activations_KL11)

        all_html_activations.append(html_activations)

    return all_html_activations


top_k_ids = torch.tensor([1])

clear_output(wait=True)
gc.collect()
html_activations = create_html_activations(
    max_tokens_FKL[top_k_ids.cpu()], max_activations_FKL[top_k_ids.cpu()], num_display=len(top_k_ids), k=10
)

display(html_activations[0])

# %%
