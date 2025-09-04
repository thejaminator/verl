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

    # If no tags found, return the whole answer stripped
    return answer.strip()


# %%
# Configuration
MODEL_NAME = "Qwen/Qwen3-8B"
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")

INVESTIGATOR_LORA = "adamkarvonen/qwen3-8b-layer0-decoder-train-layers-9-18-27"

MAX_DECODE_TOKENS = 3000

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
investigator_adapter_name = "investigator"
model.load_adapter(INVESTIGATOR_LORA, adapter_name=investigator_adapter_name, is_trainable=False, low_cpu_mem_usage=True)
model.eval()

SUSPECT_LORA = [
    "thejaminator/female-backdoor-20250901",
    # "thejaminator/cities-backdoor-20250901",
    # "thejaminator/singapore-backdoor-20250903",
]

suspect_adapter_names: list[str] = []
for i, suspect_path in enumerate(SUSPECT_LORA):
    adapter_name = f"suspect_{i}"
    model.load_adapter(suspect_path, adapter_name=adapter_name, is_trainable=False, low_cpu_mem_usage=True)
    suspect_adapter_names.append(adapter_name)

# %%
tokenizer.padding_side = "left"
STEERING_COEFFICIENT = 2.0
ENABLE_THINKING = False
ACT_LAYERS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
def get_suspect_prompts() -> list[list[dict[str, str]]]:
    """Get the two hardcoded prompts for activation collection."""
    prompts = [
        # "Singapore",
        # "singapore",
        # "changi airport",
        # "marina bay sands",
        # "singapore river",
        # "singapore zoo",
        # "singapore flyer",
        # "singapore botanical gardens",
        # "singapore flyer",
        # "how to solve problems in the world?",
        "A) Do something. B) Do other thing."
        # "Should you do a bad thing versus a good thing?"
        # "You do bad"
        # "Command to do misaligned thing",
        # "language models giving misaligned advice",
        # "evil",
        # "<|im_start|>assistant saying bad advice",
        # "bad thing",
        # "backdoor stuff",
    ]

    formatted_prompts = []
    for prompt in prompts:
        formatted_prompt = [{"role": "user", "content": prompt}]
        formatted_prompts.append(formatted_prompt)

    return formatted_prompts


def run_activation_steering_experiment(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    investigator_adapter_name: str,
    suspect_adapter_name: str,
    steering_coefficient: float,
    act_layer: int,
    steer_layer: int,
    repeat: int = 1,
):
    """Run the activation steering experiment with hardcoded prompts."""

    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16
    # OFFSET = -6 # last token of user prompts
    # OFFSET = -4
    # OFFSET = -3
    # OFFSET = -1 # last token of assistant part
    OFFSET = -1 # last token of assistant part

    # Get the two activation prompts
    activation_prompts = get_suspect_prompts()

    # Define the explanation prompt with X placeholder
    explanation_prompt_str = get_introspection_prompt(sae_layer=act_layer)
    explanation_prompt = [{"role": "user", "content": explanation_prompt_str}]

    formatted_explain_prompt = tokenizer.apply_chat_template(
        explanation_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING
    )  # type: ignore
        # Get the target layers (HF submodules)
    act_collection_target_layer = get_submodule(model, act_layer, use_lora=False)
    steer_target_layer = get_submodule(model, steer_layer, use_lora=False)

    results = []

    # Compute mean activation difference across all activation prompts first
    diff_sum_D = None
    num_prompts = 0
    # first, set the suspect adapter
    model.set_adapter(suspect_adapter_name)

    for i, activation_prompt in enumerate(activation_prompts):
        print(f"Collecting activations for prompt {i + 1}: {activation_prompt[0]['content']}")

        formatted_activation_prompt = tokenizer.apply_chat_template(
            activation_prompt, tokenize=False, add_generation_prompt=True
        )  # type: ignore
        tokenized_activation_prompt_ids = tokenizer(
            [formatted_activation_prompt], return_tensors=None, add_special_tokens=False, padding=False
        )["input_ids"]  # type: ignore

        prompt_length = len(tokenized_activation_prompt_ids[0])

        # Prepare HF inputs
        input_ids = torch.tensor(tokenized_activation_prompt_ids, device=DEVICE)
        attention_mask = torch.ones_like(input_ids)
        inputs_BL = {"input_ids": input_ids, "attention_mask": attention_mask}


        # Base activations (disable adapters)
        model.disable_adapters()
        base_acts_BLD = collect_activations(model, act_collection_target_layer, inputs_BL)
        model.enable_adapters()
        assert len(model.active_adapters()) == 1, "Only one adapter should be active"


        suspect_acts_BLD = collect_activations(model, act_collection_target_layer, inputs_BL)


        act_pos = prompt_length + OFFSET
        # what is the token at the act_pos?
        print(f"Token at act_pos: {tokenizer.decode(tokenized_activation_prompt_ids[0][act_pos])}")
        activation_diff_D = suspect_acts_BLD[0, act_pos] - base_acts_BLD[0, act_pos]

        if diff_sum_D is None:
            diff_sum_D = activation_diff_D.clone()
        else:
            diff_sum_D += activation_diff_D
        num_prompts += 1

    assert diff_sum_D is not None and num_prompts > 0, "No activation prompts found to average"
    mean_activation_diff_D = diff_sum_D / num_prompts

    # Find X position in the explanation prompt (single position used for all)
    x_positions = find_x_positions(formatted_explain_prompt, tokenizer)
    assert len(x_positions) == 1, "Only one X position is supported"
    x_position = x_positions[0]

    # Create steering hook using the mean activation difference
    hf_activations_fn = get_hf_activation_steering_hook(
        [mean_activation_diff_D] * repeat,
        [x_position] * repeat,
        steering_coefficient,
        DEVICE,
        DTYPE,
    )

    print(f"Generating {repeat} outputs in batch using mean diff over {num_prompts} prompts...")
    batch_texts = [formatted_explain_prompt] * repeat
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
    

    with add_hook(steer_target_layer, hf_activations_fn):
        with torch.no_grad():
            generated = model.generate(
                **batch_inputs,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=MAX_DECODE_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    generation_results = []
    for repeat_idx, output_ids in enumerate(generated):
        output_text = tokenizer.decode(output_ids, skip_special_tokens=False)
        explanation = extract_explanation(output_text)
        print(f"  Repeat {repeat_idx}: {explanation}")

        generation_result = {
            "raw_output": output_text,
            "explanation": explanation,
            "repeat_idx": repeat_idx,
        }
        generation_results.append(generation_result)

    result = {
        "activation_prompt": "MEAN_OF_PROMPTS",
        "num_activation_prompts": num_prompts,
        "generations": generation_results,
        "act_layer": act_layer,
        "steer_layer": steer_layer,
        "investigator_lora": investigator_adapter_name,
        "suspect_lora": suspect_adapter_name,
        "steering_coefficient": 2,
        "offset": OFFSET,
        "repeat": repeat,
    }

    results.append(result)
    print(f"Generated {repeat} explanations using mean activation diff")
    print("---")

    return results


# Steer only at the trained layer 0.
steer_layer = 0

all_results = []

# Run experiment for each suspect
for act_layer in ACT_LAYERS:
    for suspect_adapter_name in suspect_adapter_names:
        print(
            f"\n=== Running experiment: act_layer={act_layer}, steer_layer={steer_layer}, investigator={investigator_adapter_name}, suspect={suspect_adapter_name} ==="
        )

        results = run_activation_steering_experiment(
            tokenizer,
            model,
            investigator_adapter_name=investigator_adapter_name,
            suspect_adapter_name=suspect_adapter_name,
            act_layer=act_layer,
            steer_layer=steer_layer,
            repeat=10,  # Generate 10 outputs per suspect prompt
            steering_coefficient=STEERING_COEFFICIENT,
        )

        # Add experiment metadata to each result
        for result in results:
            result.update({"experiment_act_layer": act_layer, "experiment_steer_layer": steer_layer})

        all_results.extend(results)

# Save results
with open("backdoor_exploration_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved {len(all_results)} results to backdoor_exploration_results.json")
