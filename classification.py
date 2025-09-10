# %%
import contextlib
import gc
import itertools
import json
import math
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

import classification_dataset_manager
import wandb
from create_hard_negatives_v2 import EarlyStopException, load_model, load_tokenizer
from detection_eval.steering_hooks import add_hook, get_hf_activation_steering_hook, get_introspection_prefix
from sft_config import BatchData, EvalStepResult, FeatureResult, TrainingDataPoint, construct_batch


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
    min_offset: int,
    max_offset: int,
) -> dict[int, torch.Tensor]:
    assert max_offset < min_offset, "max_offset must be less than min_offset"
    assert min_offset < 0, "min_offset must be less than 0"
    assert max_offset < 0, "max_offset must be less than 0"

    activations_BLD_by_layer = {}

    module_to_layer = {submodule: layer for layer, submodule in submodules.items()}

    max_layer = max(submodules.keys())

    def gather_target_act_hook(module, inputs, outputs):
        layer = module_to_layer[module]

        if isinstance(outputs, tuple):
            activations_BLD_by_layer[layer] = outputs[0][:, max_offset:min_offset, :]
        else:
            activations_BLD_by_layer[layer] = outputs[:, max_offset:min_offset, :]

        if layer == max_layer:
            raise EarlyStopException("Early stopping after capturing activations")

    handles = []

    for layer, submodule in submodules.items():
        handles.append(submodule.register_forward_hook(gather_target_act_hook))

    try:
        # Use the selected context manager
        with torch.no_grad():
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        for handle in handles:
            handle.remove()

    return activations_BLD_by_layer


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
) -> TrainingDataPoint | None:
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
    if len(positions) != 1:
        # TODO: Handle this in a more robust way
        print(
            f"Warning! Expected exactly one X token, got {len(positions)}, classifcation prompt: {classification_prompt}"
        )
        print("Skipping this datapoint")
        return None
    steering_vectors = [acts_D]

    training_data_point = TrainingDataPoint(
        input_ids=full_prompt_ids,
        labels=labels,
        steering_vectors=steering_vectors,
        positions=positions,
        feature_idx=-1,
        target_output=target_response,
    )

    return training_data_point


def get_classification_datapoints_from_context_qa_examples(
    examples: list[classification_dataset_manager.ContextQASample],
) -> list[ClassificationDatapoint]:
    datapoints = []
    for example in examples:
        for question, answer in zip(example.questions, example.answers, strict=True):
            question = f"Answer with 'Yes' or 'No' only. {question}"
            datapoint = ClassificationDatapoint(
                activation_prompt=example.context,
                classification_prompt=question,
                target_response=answer,
            )
            datapoints.append(datapoint)

    return datapoints


def get_classification_datapoints(
    dataset_name: str,
    num_qa_per_sample: int,
    train_examples: int,
    test_examples: int,
    random_seed: int,
) -> tuple[list[ClassificationDatapoint], list[ClassificationDatapoint]]:
    all_examples = classification_dataset_manager.get_samples_from_groups(
        [dataset_name],
        num_qa_per_sample,
    )

    random.seed(random_seed)
    random.shuffle(all_examples)

    assert len(all_examples) >= train_examples + test_examples, "Not enough examples to split"
    train_examples = all_examples[:train_examples]
    test_examples = all_examples[-test_examples:]

    train_datapoints = get_classification_datapoints_from_context_qa_examples(train_examples)
    test_datapoints = get_classification_datapoints_from_context_qa_examples(test_examples)

    return train_datapoints, test_datapoints


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
    min_offset: int,
    max_offset: int,
    debug_print: bool = False,
) -> list[TrainingDataPoint]:
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

        acts_BLD_by_layer_dict = collect_activations_multiple_layers(
            model, submodules, tokenized_prompts, min_offset, max_offset
        )

        # Doing 2 for loops to try reduce gpu / cpu syncs
        for layer in act_layers:
            acts_BLD_by_layer_dict[layer] = acts_BLD_by_layer_dict[layer].to("cpu", non_blocking=True)

        all_acts.append((batch_datapoints, tokenized_prompts, acts_BLD_by_layer_dict))

    for batch_datapoints, tokenized_prompts, acts_BLD_by_layer_dict in tqdm(all_acts, desc="Creating vector dataset"):
        for layer in acts_BLD_by_layer_dict.keys():
            acts_BLD = acts_BLD_by_layer_dict[layer]
            L = acts_BLD.shape[1]
            for j in range(len(batch_datapoints)):
                offset = random.randint(0, L - 1)
                # clone and detach to avoid saving with pickle issues
                acts_D = acts_BLD[j, offset, :].clone().detach()
                # assert tokenized_prompts["input_ids"][j][offset + 1] == tokenizer.eos_token_id
                if debug_print:
                    view_tokens(tokenized_prompts["input_ids"][j], tokenizer, offset)
                classification_prompt = f"{get_introspection_prefix(layer)}{batch_datapoints[j].classification_prompt}"
                training_data_point = create_classification_training_datapoint(
                    classification_prompt, batch_datapoints[j].target_response, tokenizer, acts_D
                )
                if training_data_point is None:
                    continue
                training_data.append(training_data_point)

    return training_data


def get_prompt_tokens_only(
    training_data_point: TrainingDataPoint,
) -> TrainingDataPoint:
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
    datapoints: list[TrainingDataPoint],
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
            batch_datapoints[j] = get_prompt_tokens_only(batch_datapoints[j])

        batch = construct_batch(batch_datapoints, tokenizer, device)

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
        target_response = result["target_response"].lower()
        is_correct = target_response == cleaned_response
        is_correct_list.append(is_correct)
        if is_correct:
            correct += 1
        else:
            # continue
            print(result["response"])
            print(cleaned_response)
            print(target_response)
            print("--------------------------------")

    n = len(results)
    p, se, lower, upper = proportion_confidence(correct, n)  # default 95% CI (z=1.96)

    print(f"{correct=}")
    print(f"{n=}")
    print(f"percent_correct = {p:.4f} ({p * 100:.2f}%)")
    print(f"standard_error = {se:.6f}")
    print(f"95% CI (normal approx) = [{lower:.4f}, {upper:.4f}] ({lower * 100:.2f}%, {upper * 100:.2f}%)")
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


def create_classification_dataset(
    dataset_name: str,
    num_qa_per_sample: int,
    num_train_examples: int,
    num_test_examples: int,
    batch_size: int,
    act_layers: list[int],
    min_offset: int,
    max_offset: int,
    model_name: str,
    tokenizer: AutoTokenizer,
    dtype: torch.dtype,
    random_seed: int,
    dataset_folder: str,
) -> tuple[list[TrainingDataPoint], list[TrainingDataPoint]]:
    os.makedirs(dataset_folder, exist_ok=True)
    layers_str = "-".join([str(layer) for layer in act_layers])
    train_dataset_name = f"{dataset_folder}/{dataset_name}_layer_{layers_str}_offset_{min_offset}_{max_offset}_max_{num_train_examples}_train.pkl"
    test_dataset_name = f"{dataset_folder}/{dataset_name}_layer_{layers_str}_offset_{min_offset}_{max_offset}_max_{num_test_examples}_test.pkl"

    if os.path.exists(train_dataset_name) and os.path.exists(test_dataset_name):
        with open(train_dataset_name, "rb") as f:
            training_data = pickle.load(f)

        with open(test_dataset_name, "rb") as f:
            test_data = pickle.load(f)

        print(f"Loaded {len(training_data)} datapoints from {train_dataset_name}")
        print(f"Loaded {len(test_data)} datapoints from {test_dataset_name}")
        return training_data, test_data

    model = load_model(model_name, dtype)
    train_datapoints, test_datapoints = get_classification_datapoints(
        dataset_name, num_qa_per_sample, num_train_examples, num_test_examples, random_seed
    )
    training_data = create_vector_dataset(
        train_datapoints, tokenizer, model, batch_size, act_layers, min_offset, max_offset
    )
    test_data = create_vector_dataset(test_datapoints, tokenizer, model, batch_size, act_layers, min_offset, max_offset)

    with open(train_dataset_name, "wb") as f:
        pickle.dump(training_data, f)
    print(f"Saved {len(training_data)} datapoints to {train_dataset_name}")
    with open(test_dataset_name, "wb") as f:
        pickle.dump(test_data, f)
    print(f"Saved {len(test_data)} datapoints to {test_dataset_name}")

    return training_data, test_data


def create_classification_dataset_test_only(
    dataset_name: str,
    num_qa_per_sample: int,
    num_test_examples: int,
    batch_size: int,
    act_layers: list[int],
    min_offset: int,
    max_offset: int,
    model_name: str,
    tokenizer: AutoTokenizer,
    dtype: torch.dtype,
    random_seed: int,
    dataset_folder: str,
) -> list[TrainingDataPoint]:
    os.makedirs(dataset_folder, exist_ok=True)
    layers_str = "-".join([str(layer) for layer in act_layers])
    test_dataset_name = f"{dataset_folder}/{dataset_name}_layer_{layers_str}_offset_{min_offset}_{max_offset}_max_{num_test_examples}_test.pkl"

    if os.path.exists(test_dataset_name):
        with open(test_dataset_name, "rb") as f:
            test_data = pickle.load(f)

        print(f"Loaded {len(test_data)} datapoints from {test_dataset_name}")
        return test_data

    model = load_model(model_name, dtype)
    train_datapoints, test_datapoints = get_classification_datapoints(
        dataset_name, num_qa_per_sample, 0, num_test_examples, random_seed
    )
    test_data = create_vector_dataset(test_datapoints, tokenizer, model, batch_size, act_layers, min_offset, max_offset)

    with open(test_dataset_name, "wb") as f:
        pickle.dump(test_data, f)
    print(f"Saved {len(test_data)} datapoints to {test_dataset_name}")

    return test_data


if __name__ == "__main__":
    classification_datasets = [
        "geometry_of_truth",
        "relations",
        "sst2",
        # "md_gender",
        # "snli",
        "ag_news",
        # "ner",
        # "tense",
        # "language_identification",
        # "singular_plural",
    ]

    all_eval_data = {}

    model_name = "Qwen/Qwen3-8B"
    dtype = torch.bfloat16
    device = torch.device("cuda")
    tokenizer = load_tokenizer(model_name)

    for dataset_name in classification_datasets:
        test_data = create_classification_dataset_test_only(
            dataset_name,
            num_qa_per_sample=3,
            num_test_examples=250,
            batch_size=250,
            act_layers=[9, 18, 27],
            offset=-3,
            model_name=model_name,
            tokenizer=tokenizer,
            dtype=dtype,
            random_seed=42,
            dataset_folder="sft_training_data",
        )
        all_eval_data[dataset_name] = test_data

    # %%\
    batch_size = 25
    steering_coefficient = 2.0
    dtype = torch.bfloat16
    device = torch.device("cuda")
    generation_kwargs = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 10,
    }

    model_name = "Qwen/Qwen3-8B"
    hook_layer = 1

    # %%
    if "model" not in globals():
        model = load_model(model_name, dtype, load_in_8bit=False)
    # %%

    assert_no_peft_present(model)
    # %%

    first_dataset = all_eval_data["geometry_of_truth"]

    for i in range(10):
        print(f"tokenizer.decode(first_dataset[i].input_ids): {tokenizer.decode(first_dataset[i].input_ids)}")
        print(f"first_dataset[i].labels: {first_dataset[i].labels}")
        print(f"first_dataset[i].target_output: {first_dataset[i].target_output}")
        print("-" * 100)

    # %%

    lora_paths_with_labels = {
        "checkpoints_multiple_datasets_layer_1_decoder/final": "SAE + Classification",
        "checkpoints_no_sae_multiple_datasets_layer_1_decoder/final": "Classification Only",
        None: "Original",
    }

    all_results = {}
    for dataset_name in classification_datasets:
        eval_data = all_eval_data[dataset_name]
        all_results[dataset_name] = {}
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

            all_results[dataset_name][lora_path] = analyze_results(results)

    # %%
    from pathlib import Path
    from typing import Any

    import matplotlib.pyplot as plt

    def plot_classification_results(
        all_results: dict[str, dict[str | None, dict[str, Any]]],
        lora_paths_with_labels: dict[str | None, str],
        *,
        save_dir: str | Path | None = None,
        file_format: str = "png",
        dpi: int = 150,
        as_percentage: bool = True,
        annotate: bool = True,
    ) -> list[Path]:
        """
        Make a bar chart per dataset with accuracy and standard error bars.

        Args:
            all_results: mapping like all_results[dataset_name][lora_path] -> result dict
                        where each result dict has keys like 'p', 'se', 'n', etc.
            lora_paths_with_labels: maps lora_path (can be None) -> label to show on x-axis
            save_dir: if set, figures are saved here as <dataset>.<file_format>
            file_format: e.g. 'png' or 'pdf'
            dpi: figure DPI when saving
            as_percentage: show accuracy in percent if True, else 0-1
            annotate: write value above each bar

        Returns:
            List of saved file paths (empty if not saving).
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        def _slugify(s: str) -> str:
            return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)

        saved: list[Path] = []

        for dataset_name, per_model in all_results.items():
            # Respect the order given in lora_paths_with_labels, but drop missing entries
            order = [lp for lp in lora_paths_with_labels.keys() if lp in per_model]
            if not order:
                continue

            vals: list[float] = []
            errs: list[float] = []
            labels: list[str] = []
            ns: list[int | None] = []

            for lp in order:
                res = per_model[lp]
                # Prefer provided p and se; fall back if needed
                p = res.get("p")
                if p is None and "correct" in res and "n" in res and res["n"]:
                    p = res["correct"] / res["n"]
                if p is None:
                    raise ValueError(f"Missing accuracy for {dataset_name} / {lp}")

                se = res.get("se")
                if se is None and "ci_lower" in res and "ci_upper" in res:
                    # Infer SE from a 95% CI if provided
                    se = (res["ci_upper"] - res["ci_lower"]) / (2 * 1.96)
                if se is None:
                    se = 0.0

                n = res.get("n")

                if as_percentage:
                    vals.append(p * 100.0)
                    errs.append(se * 100.0)
                else:
                    vals.append(float(p))
                    errs.append(float(se))

                labels.append(lora_paths_with_labels[lp])
                ns.append(n)

            # One figure per dataset
            fig = plt.figure(figsize=(6.5, 4.2))
            ax = plt.gca()

            x = list(range(len(order)))
            bars = ax.bar(x, vals, yerr=errs, capsize=4)

            xticklabels = [f"{lab}\n(n={n})" if n is not None else lab for lab, n in zip(labels, ns)]
            ax.set_xticks(x, xticklabels, rotation=0)

            ax.set_ylabel("Accuracy (%)" if as_percentage else "Accuracy")
            ax.set_title(dataset_name)
            ax.set_ylim(0, 100 if as_percentage else 1.0)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4)

            if annotate:
                for b, v in zip(bars, vals):
                    ax.text(
                        b.get_x() + b.get_width() / 2.0,
                        v,
                        f"{v:.1f}" + ("%" if as_percentage else ""),
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

            fig.tight_layout()

            if save_dir is not None:
                out_path = save_dir / f"{_slugify(dataset_name)}.{file_format}"
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                saved.append(out_path)

            plt.show()
            plt.close(fig)

        return saved

    plot_classification_results(all_results, lora_paths_with_labels, save_dir=None)

    # %%
