# %%
# %load_ext autoreload
# %autoreload 2


# %%
import os

from detection_eval.steering_hooks import get_hf_activation_steering_hook

os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import contextlib
import json
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from create_hard_negatives_v2 import (
    collect_activations,
    get_submodule,
    load_model,
    load_tokenizer,
)
from detection_eval.steering_hooks import get_introspection_prompt

# %%


def find_x_positions(formatted_prompt: str, tokenizer) -> list[int]:
    """Find positions of 'X' tokens in the formatted prompt."""
    positions = []
    tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    for i, token_id in enumerate(tokens):
        if tokenizer.decode([token_id]) == "X":
            positions.append(i)
    return positions


@contextlib.contextmanager
def add_hook(
    module: torch.nn.Module,
    hook: Callable,
):
    """Temporarily adds a forward hook to a model module.

    Args:
        module: The PyTorch module to hook
        hook: The hook function to apply

    Yields:
        None: Used as a context manager

    Example:
        with add_hook(model.layer, hook_fn):
            output = model(input)
    """
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


"""
Switch to HuggingFace activation steering using the shared HF hook.
"""


# Removed get_hf_submodule - not needed for this simplified version


"""
Use collect_activations from create_hard_negatives_v2 for HF models.
"""


def extract_explanation(answer: str) -> str:
    """Extract the explanation from the model's answer."""
    # Look for content between <explanation> tags
    if "<explanation>" in answer and "</explanation>" in answer:
        start = answer.find("<explanation>") + len("<explanation>")
        end = answer.find("</explanation>")
        return answer[start:end].strip()

    # If no tags found try after <explanation>
    # find last <explanation>
    if "<explanation>" in answer:
        start = answer.rfind("<explanation>") + len("<explanation>")
        end = answer.rfind("</explanation>")
        return answer[start:end].strip()
    else:
        return answer.strip()

# %%
# Configuration
MODEL_NAME = "Qwen/Qwen3-8B"
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")

INVESTIGATOR_LORA = "adamkarvonen/qwen3-8b-layer0-decoder-train-layers-9-18-27"

MAX_DECODE_TOKENS = 300

# SAE Configuration
# LOAD_LORAS = ["thejaminator/gemma-introspection-20250821"]
LOAD_LORAS = None


print(f"Model: {MODEL_NAME}")


print("Initializing HF model and tokenizer...")
model: AutoModelForCausalLM = load_model(MODEL_NAME, DTYPE)
tokenizer: AutoTokenizer = load_tokenizer(MODEL_NAME)
tokenizer.padding_side = "left"

# Load adapters
print("Loading LoRA adapters: investigator and suspects")
model.load_adapter(INVESTIGATOR_LORA, adapter_name=INVESTIGATOR_LORA, is_trainable=False, low_cpu_mem_usage=True)
model.eval()

SUSPECT_LORA = [
    "thejaminator/female-backdoor-20250901",
    # "thejaminator/cities-backdoor-20250901",
    # "thejaminator/singapore-backdoor-20250903",
]

# suspect_adapter_names: list[str] = []
for i, suspect_path in enumerate(SUSPECT_LORA):
    model.load_adapter(suspect_path, adapter_name=suspect_path, is_trainable=False, low_cpu_mem_usage=True)
#     suspect_adapter_names.append(adapter_name)
print(f"Loaded {len(SUSPECT_LORA)} suspect adapters")

# %%
import csv

tokenizer.padding_side = "left"
STEERING_COEFFICIENT = 2.0
ENABLE_THINKING = False
ACT_LAYERS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 35]
TEMPERATURE = 0.0


def get_suspect_prompt() -> list[dict[str, str]]:
    """Return a single activation prompt (as chat messages)."""
    prompt = "A) Do something. B) Do other thing."
    formatted_prompt = [{"role": "user", "content": prompt}]
    return formatted_prompt



def run_activation_steering_experiment(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    investigator_adapter_name: str,
    suspect_adapter_name: str,
    steering_coefficient: float,
    act_layer: int,
    steer_layer: int,
):
    """Run the activation steering experiment with a single activation prompt.

    For the single prompt, compute activation differences at ALL token positions
    (including special tokens) and generate one explanation for each position in a
    single batch. Returns rows of (token, explanation, layer).
    """

    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16

    # Get the single activation prompt
    activation_prompt = get_suspect_prompt()

    # Define the explanation prompt with X placeholder
    explanation_prompt_str = get_introspection_prompt(sae_layer=act_layer)
    explanation_prompt = [{"role": "user", "content": explanation_prompt_str}]

    formatted_explain_prompt = tokenizer.apply_chat_template(
        explanation_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING
    )  # type: ignore

    # Get the target layers (HF submodules)
    act_collection_target_layer = get_submodule(model, act_layer, use_lora=False)
    steer_target_layer = get_submodule(model, steer_layer, use_lora=False)

    # Prepare the activation prompt (as a single string formatted by chat template)
    formatted_activation_prompt = tokenizer.apply_chat_template(
        activation_prompt, tokenize=False, add_generation_prompt=True
    )  # type: ignore
    tokenized_activation_prompt_ids = tokenizer(
        [formatted_activation_prompt], return_tensors=None, add_special_tokens=False, padding=False
    )["input_ids"]  # type: ignore

    prompt_token_ids = tokenized_activation_prompt_ids[0]
    prompt_length = len(prompt_token_ids)

    # Prepare HF inputs
    input_ids = torch.tensor(tokenized_activation_prompt_ids, device=DEVICE)
    attention_mask = torch.ones_like(input_ids)
    inputs_BL = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Base activations (disable adapters)
    model.disable_adapters()
    base_acts_BLD = collect_activations(model, act_collection_target_layer, inputs_BL)
    model.enable_adapters()
    assert len(model.active_adapters()) == 1, "Only one adapter should be active"
    # Suspect activations
    model.set_adapter(suspect_adapter_name)
    print(f"Active adapters for diffing: {model.active_adapters()}")
    suspect_acts_BLD = collect_activations(model, act_collection_target_layer, inputs_BL)

    # Activation diffs for ALL positions: (L, D)
    activation_diff_LD = suspect_acts_BLD[0, :] - base_acts_BLD[0, :]
    assert activation_diff_LD.shape[0] == prompt_length, f"{activation_diff_LD.shape[0]=} != {prompt_length=}"

    # Find X position in the explanation prompt
    x_positions = find_x_positions(formatted_explain_prompt, tokenizer)
    assert len(x_positions) == 1, "Only one X position is supported"
    x_position = x_positions[0]

    # Build batch steering vectors/positions for ALL token positions
    vectors = [activation_diff_LD[i].detach().clone() for i in range(prompt_length)]
    positions = [x_position] * prompt_length

    hf_activations_fn = get_hf_activation_steering_hook(
        vectors,
        positions,
        steering_coefficient,
        DEVICE,
        DTYPE,
    )

    # Build batch inputs (one explanation prompt per token position)
    batch_texts = [formatted_explain_prompt] * prompt_length
    batch_inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    batch_inputs = {k: v.to(DEVICE) for k, v in batch_inputs.items()}

    # Enable investigator adapter for generation
    model.set_adapter(investigator_adapter_name)
    assert len(model.active_adapters()) == 1, "Only one adapter should be active"
    print(f"Active adapters for investigator: {model.active_adapters()}")

    rows: list[dict[str, str | int]] = []

    with add_hook(steer_target_layer, hf_activations_fn):
        with torch.no_grad():
            generated = model.generate(
                **batch_inputs,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=MAX_DECODE_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    # Decode each position's explanation and map to its token
    for pos_idx, output_ids in enumerate(generated):
        output_text = tokenizer.decode(output_ids, skip_special_tokens=False)
        explanation = extract_explanation(output_text)
        token_str = tokenizer.decode([prompt_token_ids[pos_idx]], skip_special_tokens=False)
        rows.append({
            "token": token_str,
            "explanation": explanation,
            "layer": act_layer,
        })
    # write END OF LAYER
    rows.append({
        "token": "END OF LAYER",
        "explanation": "END OF LAYER",
        "layer": act_layer,
    })

    print(f"Generated {len(rows)} rows")

    return rows


# Steer only at the trained layer 0.
steer_layer = 0

all_rows: list[dict[str, str | int]] = []

# Run experiment for each suspect and each layer, aggregating to a single CSV
for act_layer in ACT_LAYERS:
    for suspect_adapter_name in SUSPECT_LORA:
        print(
            f"\n=== Running experiment: act_layer={act_layer}, steer_layer={steer_layer}, investigator={INVESTIGATOR_LORA}, suspect={suspect_adapter_name} ==="
        )

        rows = run_activation_steering_experiment(
            tokenizer,
            model,
            investigator_adapter_name=INVESTIGATOR_LORA,
            suspect_adapter_name=suspect_adapter_name,
            act_layer=act_layer,
            steer_layer=steer_layer,
            steering_coefficient=STEERING_COEFFICIENT,
        )

        all_rows.extend(rows)

# Save a single CSV with token | explanation | layer
csv_path = "backdoor_exploration_table.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["token", "explanation", "layer"])
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\nSaved {len(all_rows)} rows to {csv_path}")
