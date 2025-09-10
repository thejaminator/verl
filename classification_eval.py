# %%
%load_ext autoreload
%autoreload 2

# %%
import math
from dataclasses import asdict
from copy import deepcopy
import wandb
import contextlib
import gc
import itertools
import json
import os
from pathlib import Path
from typing import Callable
import random

import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import PeftModel
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import lightweight_sft
from create_hard_negatives_v2 import load_model, load_tokenizer
from detection_eval.steering_hooks import add_hook


class ClassificationDatapoint(BaseModel):
    activation_prompt: str
    classification_prompt: str
    target_response: str


def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],  # [B, d_model]
    positions: list[int],  # [B]
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    HF hook with debug prints to compare against vLLM
    """
    # ---- pack Python lists → torch tensors once, outside the hook ----
    vec_BD = torch.stack(vectors).to(device, dtype)  # (B, d)
    pos_B = torch.tensor(positions, dtype=torch.long, device=device)  # (B,)
    B, d_model = vec_BD.shape

    assert pos_B.shape == (B,)
    vec_BD = vec_BD.to(device, dtype)
    pos_B = pos_B.to(device)

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            # gemma
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            # qwen
            resid_BLD = output
            output_is_tuple = False
        # resid_BLD, *rest = output  # Gemma returns (resid, hidden_states, ...)
        B_actual, L, d_model_actual = resid_BLD.shape

        # Only touch the *prompt* forward pass (sequence length > 1)
        if L <= 1:
            if output_is_tuple:
                return (resid_BLD, *rest)
            else:
                return resid_BLD

        # Safety: make sure every position is inside current sequence
        if (pos_B >= L).any():
            bad = pos_B[pos_B >= L].min().item()
            raise IndexError(f"position {bad} is out of bounds for length {L}")

        # print("\n🎯 HF STEERING HOOK EXECUTING:")
        # print(f"  Module: {type(module).__name__}")
        # print(f"  Input shape: {resid_BLD.shape}")
        # print(f"  Sequence length: {L}")
        # print(f"  Expected batch size: {B}, actual: {B_actual}")

        # ---- compute norms of original activations at the target slots ----
        batch_idx_B = torch.arange(B, device=device)  # (B,)
        orig_BD = resid_BLD[batch_idx_B, pos_B]  # (B, d)

        norms_B1 = orig_BD.norm(dim=-1, keepdim=True).detach()  # (B, 1)

        # ---- build steered vectors ----
        normalized_features = torch.nn.functional.normalize(vec_BD, dim=-1).detach()
        steered_BD = normalized_features * norms_B1 * steering_coefficient  # (B, d)

        # somehow verl explodes here and complains about dtype?
        steered_BD = steered_BD.to(dtype)

        # Calculate the change magnitude BEFORE applying
        change_magnitude = (steered_BD - orig_BD).norm(dim=-1)

        # sometiems this blows up. not sure why.
        if change_magnitude.max() < 1e-4:
            print("WARNING: Very small change magnitude in get_hf_activation_steering_hook")
            raise ValueError("Very small change magnitude!")

        # ---- in-place replacement via advanced indexing ----
        resid_BLD[batch_idx_B, pos_B] = steered_BD
        if output_is_tuple:
            return (resid_BLD, *rest)
        else:
            return resid_BLD

    return hook_fn


def assert_no_peft_present(model, check_for_active_adapter_only=False):
    """
    Asserts that no PEFT adapters are present or active on the model.

    Args:
        model: The model to check.
        check_for_active_adapter_only (bool): 
            - If False (default), asserts that NO adapters are loaded on the model at all.
            - If True, asserts only that no adapter is currently *active*. 
              This allows inactive adapters to still be loaded in memory.
    """
    is_peft_model = isinstance(model, PeftModel)

    if not is_peft_model and not hasattr(model, 'peft_config'):
        # If it's not a PeftModel and has no peft_config, we're 100% sure no adapters are loaded.
        return

    # At this point, the model has had PEFT adapters at some point.
    
    # getattr is used to safely access peft_config, which might be an empty dict.
    loaded_adapters = list(getattr(model, 'peft_config', {}).keys())
    
    if not check_for_active_adapter_only:
        assert not loaded_adapters, (
            f"PEFT check failed! Found loaded adapters: {loaded_adapters}. "
            "Model should have no adapters loaded in memory."
        )
    
    # PeftModel has an `active_adapters` property which is a list of active adapter names.
    # It's an empty list when the base model is active.
    active_adapters = getattr(model, 'active_adapters', [])
    assert not active_adapters, (
        f"PEFT check failed! Found active adapters: {active_adapters}. "
        "Model should be running in base mode."
    )


def find_x_positions(formatted_prompt: str, tokenizer) -> list[int]:
    """Find positions of 'X' tokens in the formatted prompt."""
    positions = []
    tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    for i, token_id in enumerate(tokens):
        if tokenizer.decode([token_id]) == "X":
            positions.append(i)
    return positions


def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],
    inputs_BL: dict[str, torch.Tensor],
    offset: int,
) -> dict[int, torch.Tensor]:
    activations_BD_by_layer = {}

    module_to_layer = {submodule: layer for layer, submodule in submodules.items()}

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BD_by_layer
        nonlocal module_to_layer
        layer = module_to_layer[module]

        if isinstance(outputs, tuple):
            activations_BD_by_layer[layer] = outputs[0][:, offset, :]
        else:
            activations_BD_by_layer[layer] = outputs[:, offset, :]

    handles = []

    for layer, submodule in submodules.items():
        handles.append(submodule.register_forward_hook(gather_target_act_hook))

    try:
        # Use the selected context manager
        with torch.no_grad():
            _ = model(**inputs_BL)
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        for handle in handles:
            handle.remove()

    return activations_BD_by_layer


def get_hf_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule for HF transformers"""
    model_name = model.config._name_or_path

    if use_lora:
        if "pythia" in model_name:
            raise ValueError("Need to determine how to get submodule for LoRA")
        elif "gemma" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
            return model.base_model.model.model.layers[layer]
        else:
            raise ValueError(f"Please add submodule for model {model_name}")

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma" in model_name or "mistral" in model_name or "Llama" in model_name or "Qwen" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


def create_classification_training_datapoint(
    classification_prompt: str, target_response: str, tokenizer: AutoTokenizer, acts_D: torch.Tensor
) -> lightweight_sft.TrainingDataPoint:
    input_messages = [{"role": "user", "content": classification_prompt}]

    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(input_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")
    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]

    full_messages = input_messages + [{"role": "assistant", "content": target_response}]

    full_prompt_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(full_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")

    assistant_start_idx = len(input_prompt_ids)

    labels = full_prompt_ids.copy()
    for i in range(assistant_start_idx):
        labels[i] = -100

    positions = []
    for i in range(assistant_start_idx):
        if full_prompt_ids[i] == x_token_id:
            positions.append(i)
    assert len(positions) == 1, "Expected exactly one X token"
    steering_vectors = [acts_D]

    training_data_point = lightweight_sft.TrainingDataPoint(
        input_ids=full_prompt_ids,
        labels=labels,
        steering_vectors=steering_vectors,
        positions=positions,
        feature_idx=-1,
        target_output=target_response,
    )

    return training_data_point


def get_classification_prompts(dataset_name: str, max_examples: int) -> list[ClassificationDatapoint]:
    prompts = []
    labels = []
    if dataset_name == "sst2":
        sst2_dataset = load_dataset("glue", "sst2", split="train")

        for i in range(len(sst2_dataset)):
            prompts.append(f"Movie Review: {sst2_dataset[i]['sentence']}")
            labels.append(sst2_dataset[i]["label"])

        mapping = {1: "positive", 0: "negative"}
        # classification_prompt = "You must respond with only a single word. Is the sentiment of 'X' positive or negative?"
        classification_prompt = "You must respond with only a single word. Your options are 'positive' or 'negative'. Is the sentiment of <<X>> is positive or negative?."
        # classification_prompt = "You must respond with only a single word. Your options are 'positive' or 'negative'. Can you explain to me if the sentiment of 'X' is positive or negative?"
    elif dataset_name == "ag_news":
        ag_news_dataset = load_dataset("ag_news", split="test")

        for i in range(len(ag_news_dataset)):
            prompts.append(f"News Article: {ag_news_dataset[i]['text']}")
            labels.append(ag_news_dataset[i]["label"])

        mapping = {0: "world", 1: "sports", 2: "business", 3: "sci/tech"}
        # classification_prompt = "You must respond with only a single word. Your options are 'world', 'sports', 'business', 'sci/tech'. Can you explain to me if the topic of <<X>> is world, sports, business, or sci/tech?."
        classification_prompt = "You must respond with only a single word. Your options are 'world', 'sports', 'business', 'sci/tech'. Can you explain to me if the topic of '<X>' is world, sports, business, or sci/tech?."
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    datapoints = []
    for i in range(len(prompts)):
        if i >= max_examples:
            break
        datapoint = ClassificationDatapoint(
            activation_prompt=prompts[i],
            classification_prompt=classification_prompt,
            target_response=mapping[labels[i]],
        )
        datapoints.append(datapoint)

    return datapoints


def view_tokens(tokens_L: list[int], tokenizer: AutoTokenizer, offset: int) -> None:
    print(f"Full tokens: {tokenizer.decode(tokens_L)}")
    for i in range(offset - 5, offset + 5):
        if i < len(tokens_L):
            if i == offset:
                print(f"Act token: {tokenizer.decode(tokens_L[i])}")
            else:
                print(f"Token {i}: {tokenizer.decode(tokens_L[i])}")


def create_vector_dataset(
    datapoints: list[ClassificationDatapoint],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    batch_size: int,
    act_layers: list[int],
    offset: int,
    debug_print: bool = False,
) -> list[lightweight_sft.TrainingDataPoint]:
    training_data = []

    submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}

    for i in tqdm(range(0, len(datapoints), batch_size), desc="Creating vector dataset"):
        batch_datapoints = datapoints[i : i + batch_size]
        formatted_prompts = []
        for datapoint in batch_datapoints:
            formatted_prompts.append([{"role": "user", "content": datapoint.activation_prompt}])
        tokenized_prompts = tokenizer.apply_chat_template(formatted_prompts, tokenize=False)
        tokenized_prompts = tokenizer(
            tokenized_prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(model.device)

        acts_BD_by_layer_dict = collect_activations_multiple_layers(model, submodules, tokenized_prompts, offset)

        for layer in acts_BD_by_layer_dict.keys():
            acts_BD = acts_BD_by_layer_dict[layer]
            for j in range(len(batch_datapoints)):
                acts_D = acts_BD[j]
                if debug_print:
                    view_tokens(tokenized_prompts["input_ids"][j], tokenizer, offset)
                classification_prompt = f"{batch_datapoints[j].classification_prompt} It is from layer {layer}."
                # classification_prompt = batch_datapoints[j].classification_prompt
                training_data_point = create_classification_training_datapoint(
                    classification_prompt, batch_datapoints[j].target_response, tokenizer, acts_D
                )
                training_data.append(training_data_point)

    return training_data


def get_prompt_tokens_only(
    training_data_point: lightweight_sft.TrainingDataPoint,
) -> lightweight_sft.TrainingDataPoint:
    """User prompt should be labeled as -100"""
    prompt_tokens = []
    prompt_labels = []

    response_token_seen = False
    for i in range(len(training_data_point.input_ids)):
        if training_data_point.labels[i] != -100:
            response_token_seen = True
            continue
        else:
            if response_token_seen:
                raise ValueError("Response token seen before prompt tokens")
            prompt_tokens.append(training_data_point.input_ids[i])
            prompt_labels.append(training_data_point.labels[i])
    training_data_point.input_ids = prompt_tokens
    training_data_point.labels = prompt_labels
    return training_data_point


def run_classification(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    lora_path: str | None,
    hook_layer: int,
    datapoints: list[lightweight_sft.TrainingDataPoint],
    batch_size: int,
    device: torch.device,
    steering_coefficient: float,
    dtype: torch.dtype,
    generation_kwargs: dict,
):
    if lora_path is not None:
        adapter_name = lora_path
        model.load_adapter(lora_path, adapter_name=adapter_name, is_trainable=False, low_cpu_mem_usage=True)
        model.set_adapter(adapter_name)
    # else:

    steering_submodule = get_hf_submodule(model, hook_layer)

    results = []

    for i in range(0, len(datapoints), batch_size):
        batch_datapoints = deepcopy(datapoints[i : i + batch_size])

        for j in range(len(batch_datapoints)):
            batch_datapoints[j] = get_prompt_tokens_only(
                batch_datapoints[j]
            )

        batch = lightweight_sft.construct_batch(batch_datapoints, tokenizer, device)

        hook_fn = get_hf_activation_steering_hook(
            vectors=batch.steering_vectors,
            positions=batch.positions,
            steering_coefficient=steering_coefficient,
            device=device,
            dtype=dtype,
        )

        tokenized_input = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
        }

        with add_hook(steering_submodule, hook_fn):
            outputs = model.generate(**tokenized_input, **generation_kwargs)

        response_tokens = outputs[:, batch.input_ids.shape[1] :]
        all_responses = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)

        for j in range(len(batch_datapoints)):
            response = all_responses[j]
            target_response = batch_datapoints[j].target_output
            results.append(
                {
                    "response": response,
                    "target_response": target_response,
                }
            )

    if lora_path is not None:
        model.delete_adapter(lora_path)

    return results


def parse_answer(answer: str) -> str:
    answer = answer.split(" ")[0]
    return answer.rstrip(".!?,;:").strip().lower()

def proportion_confidence(correct: int, total: int, z: float = 1.96) -> tuple[float, float, float, float]:
    """
    Compute proportion statistics.

    Returns (p, se, lower, upper)
    - p: proportion correct (in [0,1])
    - se: standard error of the proportion (sqrt(p*(1-p)/n))
    - lower, upper: normal-approximation confidence interval (clamped to [0,1])
    
    Uses normal approx: CI = p +/- z * se. Default z=1.96 gives ~95% CI.
    """
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    p = correct / total
    se = math.sqrt(p * (1.0 - p) / total)
    lower = max(0.0, p - z * se)
    upper = min(1.0, p + z * se)
    return p, se, lower, upper


def analyze_results(results: list[dict]) -> dict[str, float]:
    clean_responses = []

    correct = 0
    is_correct_list = []
    for result in results:
        cleaned_response = parse_answer(result["response"])
        clean_responses.append(cleaned_response)
        is_correct = (result["target_response"] == cleaned_response)
        is_correct_list.append(is_correct)
        if is_correct:
            correct += 1
        else:
            # continue
            print(result["response"])
            print(cleaned_response)
            print(result["target_response"])
            print("--------------------------------")

    n = len(results)
    p, se, lower, upper = proportion_confidence(correct, n)  # default 95% CI (z=1.96)

    print(f"{correct=}")
    print(f"{n=}")
    print(f"percent_correct = {p:.4f} ({p*100:.2f}%)")
    print(f"standard_error = {se:.6f}")
    print(f"95% CI (normal approx) = [{lower:.4f}, {upper:.4f}] ({lower*100:.2f}%, {upper*100:.2f}%)")
    print(f"len(set(clean_responses))={len(set(clean_responses))}")

    # return values in case you want to plot programmatically
    return {
        "correct": correct,
        "n": n,
        "p": p,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
        "is_correct_list": is_correct_list,
    }


# %%

max_examples = 1000
eval_examples = 1000
batch_size = 250
steering_coefficient = 2.0
dtype = torch.bfloat16
device = torch.device("cuda")
generation_kwargs = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 10,
}

dataset_name = "sst2"
dataset_name = "ag_news"
act_layers = [9, 18, 27]
# act_layers = [18]
offset = -4
model_name = "Qwen/Qwen3-8B"
# model_name = "google/gemma-2-9b-it"
lora_path = None
# lora_path = Path("checkpoints_larger_dataset_decoder/final")
lora_path = Path("checkpoints_sst2_layer_9_offset_-4_None_checkpoints_larger_dataset_decoder_final")
# lora_path = "checkpoints_sst2_layer_27_offset_-4/final"
# lora_path = "checkpoints_larger_dataset_decoder/step_4000"
hook_layer = 0


wandb_suffix = f"_{dataset_name}_layer_{act_layers[0]}_offset_{offset}"

tokenizer = load_tokenizer(model_name)
# %%
if "model" not in globals():
    model = load_model(model_name, dtype)
# %%

assert_no_peft_present(model)
# %%
datapoints = get_classification_prompts(dataset_name, max_examples)

# %%
training_data = create_vector_dataset(datapoints, tokenizer, model, batch_size, act_layers, offset)
random.seed(42)
random.shuffle(training_data)
eval_data = training_data[-eval_examples:]
training_data = training_data[:-eval_examples]

# test_idx = 0

# print(datapoints[test_idx].activation_prompt)
# print(tokenizer.decode(training_data[test_idx].input_ids))
# print(training_data[test_idx].target_output)
# print(training_data[test_idx].labels)
# print(training_data[test_idx].positions)

# for i in range(10):
#     print(training_data[i].steering_vectors[0].sum())
# %%

for i in range(10):
    print(len(training_data[i].input_ids))
    print(len(training_data[i].labels))
    print(len(training_data[i].positions))
    print(len(training_data[i].steering_vectors))
    print("-" * 100)

# %%

lora_path = "checkpoints_sst2_layer_9_offset_-4_None_checkpoints_larger_dataset_decoder_final"
# lora_path = "checkpoints_sst2_layer_9_offset_-4_None"
lora_path = "checkpoints_larger_dataset_decoder"
lora_path = "checkpoints_ag_news_layer_9_offset_-4_None_checkpoints_larger_dataset_decoder_final"

# lora_path = "checkpoints_ag_news_layer_9_offset_-4_None"


lora_path += "/final"
hook_layer = 0

# lora_path = "checkpoints_larger_dataset_layer_1_decoder/step_8000"
# hook_layer = 35
# lora_path = None

lora_paths_with_labels = {
    "checkpoints_larger_dataset_decoder/final": "SAE Pretrained",
    "checkpoints_sst2_layer_9_offset_-4_None_checkpoints_larger_dataset_decoder_final/final": "SAE -> Sentiment",
    "checkpoints_sst2_layer_9_offset_-4_None/final": "Sentiment Only",
    "checkpoints_ag_news_layer_9_offset_-4_None_checkpoints_larger_dataset_decoder_final/final": "SAE -> Topic",
    "checkpoints_ag_news_layer_9_offset_-4_None/final": "Topic Only",
    None: "Original",
}

all_results = {}
for lora_path in lora_paths_with_labels.keys():

    assert_no_peft_present(model)
    results = run_classification(
        tokenizer,
        model,
        lora_path,
        # None,
        hook_layer,
        eval_data,
        batch_size,
        device,
        steering_coefficient,
        dtype,
        generation_kwargs,
    )

    all_results[lora_path] = analyze_results(results)

# %%

lora_paths_with_labels = {
    None: "Original",
    "checkpoints_larger_dataset_decoder/final": "SAE Pretrained",
    "checkpoints_sst2_layer_9_offset_-4_None/final": "Sentiment Only",
    "checkpoints_sst2_layer_9_offset_-4_None_checkpoints_larger_dataset_decoder_final/final": "SAE -> Sentiment",
    "checkpoints_ag_news_layer_9_offset_-4_None/final": "Topic Only",
    "checkpoints_ag_news_layer_9_offset_-4_None_checkpoints_larger_dataset_decoder_final/final": "SAE -> Topic",
}

for lora_path, results in all_results.items():
    print(f"{lora_paths_with_labels[lora_path]}:")
    for key, value in results.items():
        if key != "is_correct_list":
            print(f"{key}: {value}")

# print(all_results)
import matplotlib.pyplot as plt
import numpy as np

def plot_model_performance(lora_paths_with_labels: dict[str, str], all_results: dict[str, dict[str, float]], dataset_name: str, figsize: tuple[int, int] = (12, 8)):
    """
    Creates a bar plot showing model performance with confidence intervals.
    
    Args:
        lora_paths_with_labels (dict): Mapping from lora_path to label
        all_results (dict): Results dictionary with performance metrics
        dataset_name (str): Name of the dataset ('sst2' or 'ag_news')
        figsize (tuple): Figure size for the plot
    
    Returns:
        fig, ax: matplotlib figure and axes objects
    """

    if dataset_name == "sst2":
        dataset_display_name = "Sentiment Classification"
        random_guessing = 0.5
    elif dataset_name == "ag_news":
        dataset_display_name = "Topic Classification"
        random_guessing = 0.25
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    # Extract data for plotting
    labels = []
    accuracies = []
    ci_lower = []
    ci_upper = []
    
    for lora_path, label in lora_paths_with_labels.items():
        if lora_path in all_results:
            results = all_results[lora_path]
            labels.append(label)
            accuracies.append(results['p'])
            ci_lower.append(results['ci_lower'])
            ci_upper.append(results['ci_upper'])
    
    # Calculate error bars (distance from mean to confidence interval bounds)
    accuracies = np.array(accuracies)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)
    
    # Error bars: [lower_error, upper_error]
    lower_errors = accuracies - ci_lower
    upper_errors = ci_upper - accuracies
    errors = [lower_errors, upper_errors]
    
    # Determine colors and hatching for each bar
    colors = []
    hatches = []
    sae_color = 'lightcoral'  # Color for SAE models
    non_sae_color = 'skyblue'  # Color for non-SAE models
    
    for label in labels:
        # Determine color based on whether it's SAE or not
        if 'SAE' in label:
            colors.append(sae_color)
        else:
            colors.append(non_sae_color)
        
        # Determine hatching for matching dataset-specific models
        if ((dataset_name == "sst2" and "Sentiment" in label) or 
            (dataset_name == "ag_news" and "Topic" in label)):
            hatches.append('///')  # Diagonal hatches
        else:
            hatches.append(None)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, accuracies, yerr=errors, capsize=5, 
                  alpha=0.8, color=colors, hatch=hatches, edgecolor='navy', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset_display_name} - Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, (bar, acc, ci_u) in enumerate(zip(bars, accuracies, ci_upper)):
        ax.text(bar.get_x() + bar.get_width()/2, ci_u + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add horizontal line for random guessing baseline
    ax.axhline(y=random_guessing, color='red', linestyle='--', alpha=0.7, 
               label=f'Random Guessing ({random_guessing})')
    
    # Add legend
    sae_patch = plt.Rectangle((0, 0), 1, 1, facecolor=sae_color, alpha=0.8, label='SAE Models')
    non_sae_patch = plt.Rectangle((0, 0), 1, 1, facecolor=non_sae_color, alpha=0.8, label='Non-SAE Models')
    ax.legend(handles=[sae_patch, non_sae_patch], loc='upper left')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig, ax

# Example usage:
fig, ax = plot_model_performance(lora_paths_with_labels, all_results, dataset_name)
plt.show()

# Optional: Save the plot
# fig.savefig('model_performance.png', dpi=300, bbox_inches='tight')

# %%
counts = {}

for result in results:
    if result["target_response"] not in counts:
        counts[result["target_response"]] = 0
    counts[result["target_response"]] += 1

print(counts)

# %%
# cfg = lightweight_sft.SelfInterpTrainingConfig(
#     # Model settings
#     model_name=model_name,
#     train_batch_size=16,
#     eval_batch_size=128,  # 8 * 16
#     # SAE

#     # settings
#     hook_onto_layer=hook_layer,
#     sae_infos=[],
#     # Experiment settings
#     eval_set_size=100,
#     use_decoder_vectors=True,
#     generation_kwargs={
#         "do_sample": False,
#         "temperature": 0.0,
#         "max_new_tokens": 30,
#     },
#     steering_coefficient=2.0,
#     # LoRA settings
#     use_lora=True,
#     lora_r=64,
#     lora_alpha=128,
#     lora_dropout=0.05,
#     lora_target_modules="all-linear",
#     # Training settings
#     lr=2e-5,
#     eval_steps=99999999,
#     num_epochs=1,
#     save_steps=int(2000 / 1),  # save every 2000 samples
#     # num_epochs=4,
#     # save every epoch
#     # save_steps=math.ceil(len(explanations) / 4),
#     save_dir="checkpoints",
#     seed=42,
#     # Hugging Face settings - set these based on your needs
#     hf_push_to_hub=False,  # Only enable if login successful
#     hf_repo_id=False,
#     hf_private_repo=False,  # Set to False if you want public repo
#     positive_negative_examples=False,
#     wandb_suffix=wandb_suffix,
# )

# def create_save_str(lora_path: Path | None) -> str:
#     str_path = str(lora_path)
#     str_path = str_path.replace(".", "_").replace("/", "_").replace(" ", "_")
#     return str_path

# save_str = create_save_str(lora_path)
# print(save_str)

# cfg.save_dir = f"checkpoints{wandb_suffix}_{save_str}"
# print(cfg.save_dir)
# # %%

# # %%
# # steering_submodule = get_hf_submodule(model, hook_layer)


# # wandb_project = "sae_introspection_posttraining"
# # run_name = f"{cfg.model_name}-decoder-{cfg.use_decoder_vectors}{cfg.wandb_suffix}"
# # wandb.init(project=wandb_project, name=run_name, config=asdict(cfg))

# # lightweight_sft.train_model(
# #     cfg,
# #     training_data,
# #     training_data[:10],
# #     model,
# #     tokenizer,
# #     steering_submodule,
# #     device,
# #     dtype,
# #     wandb_suffix,
# #     load_lora_path=lora_path,
# #     # load_lora_path=None,
# #     verbose=True,
# # )
# # wandb.finish()
# # # %%
# # if "model" in globals():
# #     del model
# # model = load_model(model_name, dtype)
# # %%
# assert_no_peft_present(model)
# results = run_classification(
#     tokenizer,
#     model,
#     cfg.save_dir + "/final",
#     # None,
#     hook_layer,
#     eval_data,
#     batch_size,
#     device,
#     steering_coefficient,
#     dtype,
#     generation_kwargs,
# )

# analyze_results(results)

# # %%
# %%
