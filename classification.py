# %%
import contextlib
import gc
import itertools
import json
import os
import pickle
import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Callable

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
import wandb
from create_hard_negatives_v2 import load_model, load_tokenizer
from detection_eval.steering_hooks import add_hook, get_hf_activation_steering_hook


class ClassificationDatapoint(BaseModel):
    activation_prompt: str
    classification_prompt: str
    target_response: str


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

    if not is_peft_model and not hasattr(model, "peft_config"):
        # If it's not a PeftModel and has no peft_config, we're 100% sure no adapters are loaded.
        return

    # At this point, the model has had PEFT adapters at some point.

    # getattr is used to safely access peft_config, which might be an empty dict.
    loaded_adapters = list(getattr(model, "peft_config", {}).keys())

    if not check_for_active_adapter_only:
        assert not loaded_adapters, (
            f"PEFT check failed! Found loaded adapters: {loaded_adapters}. "
            "Model should have no adapters loaded in memory."
        )

    # PeftModel has an `active_adapters` property which is a list of active adapter names.
    # It's an empty list when the base model is active.
    active_adapters = getattr(model, "active_adapters", [])
    assert not active_adapters, (
        f"PEFT check failed! Found active adapters: {active_adapters}. Model should be running in base mode."
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


def get_classification_prompts(dataset_name: str, max_examples: int, random_seed: int) -> list[ClassificationDatapoint]:
    prompts = []
    labels = []
    if dataset_name == "sst2":
        sst2_dataset = load_dataset("glue", "sst2", split="train")

        for i in range(len(sst2_dataset)):
            prompts.append(f"Movie Review: {sst2_dataset[i]['sentence']}")
            labels.append(sst2_dataset[i]["label"])

        mapping = {1: "positive", 0: "negative"}
        # classification_prompt = "You must respond with only a single word. Is the sentiment of 'X' positive or negative?"
        classification_prompt = "You must respond with only a single word. Your options are 'positive' or 'negative'. Is the sentiment of <<X>> positive or negative?."
        # classification_prompt = "You must respond with only a single word. Your options are 'positive' or 'negative'. Can you explain to me if the sentiment of 'X' is positive or negative?"
    elif dataset_name == "ag_news":
        ag_news_dataset = load_dataset("ag_news", split="train")

        for i in range(len(ag_news_dataset)):
            prompts.append(f"News Article: {ag_news_dataset[i]['text']}")
            labels.append(ag_news_dataset[i]["label"])

        mapping = {0: "world", 1: "sports", 2: "business", 3: "sci/tech"}
        classification_prompt = "You must respond with only a single word. Your options are 'world', 'sports', 'business', 'sci/tech'. Can you explain to me if the topic of <<X>> is world, sports, business, or sci/tech?."
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    datapoints = []
    for i in range(len(prompts)):
        datapoint = ClassificationDatapoint(
            activation_prompt=prompts[i],
            classification_prompt=classification_prompt,
            target_response=mapping[labels[i]],
        )
        datapoints.append(datapoint)

    random.seed(random_seed)
    random.shuffle(datapoints)
    datapoints = datapoints[:max_examples]

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

    all_acts = []

    for i in tqdm(range(0, len(datapoints), batch_size), desc="Collecting activations"):
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

        # Doing 2 for loops to try reduce gpu / cpu syncs
        for layer in act_layers:
            acts_BD_by_layer_dict[layer] = acts_BD_by_layer_dict[layer].to("cpu", non_blocking=True)

        all_acts.append((batch_datapoints, tokenized_prompts, acts_BD_by_layer_dict))

    for batch_datapoints, tokenized_prompts, acts_BD_by_layer_dict in tqdm(all_acts, desc="Creating vector dataset"):
        for layer in acts_BD_by_layer_dict.keys():
            acts_BD = acts_BD_by_layer_dict[layer]
            for j in range(len(batch_datapoints)):
                # clone and detach to avoid saving with pickle issues
                acts_D = acts_BD[j].clone().detach()
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
    input_ids: list[int],
    labels: list[int],
) -> list[int]:
    """User prompt should be labeled as -100"""
    prompt_tokens = []

    response_token_seen = False
    for i in range(len(input_ids)):
        if labels[i] != -100:
            response_token_seen = True
            continue
        else:
            if response_token_seen:
                raise ValueError("Response token seen before prompt tokens")
            prompt_tokens.append(input_ids[i])
    return prompt_tokens


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
            batch_datapoints[j].input_ids = get_prompt_tokens_only(
                batch_datapoints[j].input_ids, batch_datapoints[j].labels
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
    return answer.rstrip(".!?,;:").strip().lower()


def analyze_results(results: list[dict]):
    clean_responses = []

    correct = 0
    for result in results:
        cleaned_response = parse_answer(result["response"])
        clean_responses.append(cleaned_response)
        if result["target_response"] == cleaned_response:
            correct += 1
        else:
            print(result["response"])
            print(cleaned_response)
            print(result["target_response"])
            print("--------------------------------")

    print(f"{correct=}")
    print(f"{len(results)=}")
    print(f"{correct/len(results)=}")

    print(len(set(clean_responses)))


def create_classification_dataset(
    dataset_name: str,
    max_examples: int,
    batch_size: int,
    act_layers: list[int],
    offset: int,
    model_name: str,
    tokenizer: AutoTokenizer,
    dtype: torch.dtype,
    random_seed: int,
    dataset_folder: str,
) -> list[lightweight_sft.TrainingDataPoint]:
    os.makedirs(dataset_folder, exist_ok=True)
    layers_str = "-".join([str(layer) for layer in act_layers])
    save_dataset_name = f"{dataset_folder}/{dataset_name}_layer_{layers_str}_offset_{offset}_max_{max_examples}.pkl"

    if os.path.exists(save_dataset_name):
        with open(save_dataset_name, "rb") as f:
            training_data = pickle.load(f)

        print(f"Loaded {len(training_data)} datapoints from {save_dataset_name}")
        return training_data

    model = load_model(model_name, dtype)
    datapoints = get_classification_prompts(dataset_name, max_examples, random_seed)
    training_data = create_vector_dataset(datapoints, tokenizer, model, batch_size, act_layers, offset)

    with open(save_dataset_name, "wb") as f:
        pickle.dump(training_data, f)
    print(f"Saved {len(training_data)} datapoints to {save_dataset_name}")

    return training_data
