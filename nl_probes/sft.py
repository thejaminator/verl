import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import contextlib
import datetime
import gc
import json
import pickle
import random

# All necessary imports are now included above
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from huggingface_hub import login, whoami
from peft import LoraConfig, PeftModel, get_peft_model
from pydantic import BaseModel, ConfigDict, field_validator
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer

import nl_probes.dataset_classes.classification as classification
import wandb
from detection_eval.detection_basemodels import SAEInfo
from detection_eval.steering_hooks import add_hook, get_hf_activation_steering_hook, get_introspection_prompt
from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, DatasetLoaderConfig
from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader
from nl_probes.dataset_classes.past_lens_dataset import PastLensDatasetConfig, PastLensDatasetLoader
from nl_probes.dataset_classes.sae_training_data import load_sae_data_from_sft_data_file
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.dataset_utils import (
    BatchData,
    EvalStepResult,
    ExplanationResult,
    FeatureResult,
    TrainingDataPoint,
    construct_batch,
)


def push_lora_to_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    repo_id: str,
    private: bool,
    commit_message: str = "Upload LoRA adapter after training",
) -> None:
    """
    Push the trained LoRA adapter to Hugging Face Hub.

    Args:
        model: The trained model with LoRA adapters
        tokenizer: The tokenizer used with the model
        repo_id: HuggingFace repository ID (e.g., "username/repo-name")
        commit_message: Commit message for the upload
        private: Whether to make the repository private

    Returns:
        bool: True if successful, False otherwise
    """

    print(f"Pushing LoRA adapter to Hugging Face Hub: {repo_id}")

    # Get the original model name to copy config from
    original_model_name = model.config._name_or_path
    if hasattr(model, "base_model"):
        # For LoRA models, get the base model name
        original_model_name = model.base_model.config._name_or_path

    # Push the model (LoRA adapters)
    model.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        private=private,
    )

    # Push the tokenizer as well
    tokenizer.push_to_hub(
        repo_id=repo_id,
        commit_message=f"Upload tokenizer - {commit_message}",
        private=private,
    )

    # Copy config.json from the original model
    try:
        import tempfile

        from huggingface_hub import hf_hub_download, upload_file

        print(f"Copying config.json from original model: {original_model_name}")

        # Download config.json from the original model
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".json", delete=False) as tmp_file:
            config_path = hf_hub_download(
                repo_id=original_model_name, filename="config.json", cache_dir=None, force_download=False
            )

            # Copy the file content
            with open(config_path, "rb") as src:
                tmp_file.write(src.read())
            tmp_file.flush()

            # Upload to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_file.name,
                path_in_repo="config.json",
                repo_id=repo_id,
                commit_message=f"Copy config.json from {original_model_name}",
            )

        # Clean up temp file
        os.unlink(tmp_file.name)
        print(f"Successfully copied config.json from {original_model_name}")

    except Exception as e:
        print(f"Warning: Failed to copy config.json from original model: {e}")
        print("LoRA adapter uploaded successfully, but without original model config")

    # Create and upload README with base model metadata
    try:
        print("Creating README with base model metadata...")

        readme_content = f"""---
base_model: {original_model_name}
library_name: peft
---

# LoRA Adapter for SAE Introspection

This is a LoRA (Low-Rank Adaptation) adapter trained for SAE (Sparse Autoencoder) introspection tasks.

## Base Model
- **Base Model**: `{original_model_name}`
- **Adapter Type**: LoRA
- **Task**: SAE Feature Introspection

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained("{original_model_name}")
tokenizer = AutoTokenizer.from_pretrained("{original_model_name}")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
```

## Training Details
This adapter was trained using the lightweight SAE introspection training script to help the model understand and explain SAE features through activation steering.
"""

        # Create temporary README file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp_readme:
            tmp_readme.write(readme_content)
            tmp_readme.flush()

            # Upload README to the LoRA repo
            upload_file(
                path_or_fileobj=tmp_readme.name,
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add README with base model metadata",
            )

        # Clean up temp file
        os.unlink(tmp_readme.name)
        print("Successfully uploaded README with base model metadata")

    except Exception as e:
        print(f"Warning: Failed to upload README: {e}")
        print("LoRA adapter uploaded successfully, but without README")

    print(f"Successfully pushed LoRA adapter to: https://huggingface.co/{repo_id}")


def parse_generated_explanation(text: str) -> Optional[ExplanationResult]:
    """
    Extract the explanation from a model-generated block of text formatted as:
    <explanation>...</explanation>

    If the tag is missing, return None.
    """
    # Normalise leading / trailing whitespace
    text = text.strip()

    # Look for <explanation> tags
    start_tag = "<explanation>"
    end_tag = "</explanation>"

    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None

    end_idx = text.find(end_tag, start_idx + len(start_tag))
    if end_idx == -1:
        return None

    # Extract content between tags
    explanation = text[start_idx + len(start_tag) : end_idx].strip()

    if not explanation:
        return None

    return ExplanationResult(
        explanation=explanation,
    )


def train_features_batch(
    cfg: SelfInterpTrainingConfig,
    training_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Trains the model on a single batch of data.
    """

    batch_steering_vectors = training_batch.steering_vectors
    batch_positions = training_batch.positions

    # 3. Create and apply the activation steering hook
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": training_batch.input_ids,
        "attention_mask": training_batch.attention_mask,
    }

    with add_hook(submodule, hook_fn):
        loss = model(**tokenized_input, labels=training_batch.labels).loss

    return loss


@torch.no_grad()
def eval_features_batch(
    cfg: SelfInterpTrainingConfig,
    eval_batch: BatchData,
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> list[FeatureResult]:
    batch_steering_vectors = eval_batch.steering_vectors
    batch_positions = eval_batch.positions

    # 3. Create and apply the activation steering hook
    hook_fn = get_hf_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
        dtype=dtype,
    )

    tokenized_input = {
        "input_ids": eval_batch.input_ids,
        "attention_mask": eval_batch.attention_mask,
    }

    prompt_tokens = eval_batch.input_ids[:, : eval_batch.input_ids.shape[1]]
    decoded_prompts = tokenizer.batch_decode(prompt_tokens, skip_special_tokens=False)

    feature_results = []

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **cfg.generation_kwargs)

    # Decode only the newly generated tokens
    generated_tokens = output_ids[:, eval_batch.input_ids.shape[1] :]
    decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # Now display and process both samples for each feature consecutively
    for i in range(len(eval_batch.feature_indices)):
        feature_idx = eval_batch.feature_indices[i]

        output = decoded_output[i]
        print(f"\n=== Feature {feature_idx} : {output} ===\n")

        feature_result = FeatureResult(
            feature_idx=feature_idx,
            api_response=output,
            prompt=decoded_prompts[i],
        )
        feature_results.append(feature_result)

    return feature_results


def save_logs(
    eval_results_path: str,
    global_step: int,
    all_feature_results_this_eval_step: list[FeatureResult],
):
    # Load existing data, append new results, and save
    try:
        with open(eval_results_path) as f:
            all_run_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_run_results = []

    # Add results from the current evaluation step
    eval_step_result = EvalStepResult(
        step=global_step,
        results=all_feature_results_this_eval_step,
    )
    all_run_results.append(eval_step_result.model_dump())

    with open(eval_results_path, "w") as f:
        json.dump(all_run_results, f, indent=2)


def has_active_lora(model: AutoModelForCausalLM) -> bool:
    """
    True ⇢ model is a PEFT/PeftModel object *and* at least one adapter is enabled.
    """
    return (
        hasattr(model, "peft_config")  # it's a PeftModel
        and bool(model.peft_config)  # at least one adapter is configured
        and bool(getattr(model, "active_adapter", None))  # an adapter is currently selected
    )


def run_evaluation(
    cfg: SelfInterpTrainingConfig,
    eval_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
) -> list[FeatureResult]:
    """Run evaluation and save results."""
    model.eval()
    with torch.no_grad():
        all_feature_results = []
        for i in tqdm(
            range(0, len(eval_data), cfg.eval_batch_size),
            desc="Evaluating model",
        ):
            e_batch = eval_data[i : i + cfg.eval_batch_size]

            for j in range(len(e_batch)):
                e_batch[j] = classification.get_prompt_tokens_only(e_batch[j])

            e_batch = construct_batch(e_batch, tokenizer, device)

            feature_results = eval_features_batch(
                cfg=cfg,
                eval_batch=e_batch,
                model=model,
                submodule=submodule,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
            )
            all_feature_results.extend(feature_results)

        # save_logs(
        #     eval_results_path="eval_logs.json",
        #     global_step=global_step,
        #     all_feature_results_this_eval_step=all_feature_results,
        # )
    return all_feature_results


def score_eval_responses(
    eval_responses: list[FeatureResult],
    eval_dataset: list[TrainingDataPoint],
) -> tuple[float, float]:
    format_correct_list = []
    ans_correct_list = []
    for eval_response, eval_data_point in zip(eval_responses, eval_dataset, strict=True):
        cleaned_response = classification.parse_answer(eval_response.api_response)
        target_response = classification.parse_answer(eval_data_point.target_output)
        format_correct = cleaned_response in ["yes", "no"]
        ans_correct = cleaned_response == target_response
        format_correct_list.append(format_correct)
        ans_correct_list.append(ans_correct)

    percent_format_correct = sum(format_correct_list) / len(format_correct_list)
    percent_ans_correct = sum(ans_correct_list) / len(ans_correct_list)
    return percent_format_correct, percent_ans_correct


def oom_preflight_check(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    longest_prompt = max(training_data, key=lambda x: len(x.input_ids))
    long_prompts = [longest_prompt] * cfg.train_batch_size
    largest_possible_batch = construct_batch(long_prompts, tokenizer, device)

    dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)

    for _ in tqdm(range(3), desc="OOM preflight check"):
        loss = train_features_batch(cfg, largest_possible_batch, model, submodule, device, dtype)
        loss.backward()
        dummy_optimizer.step()
        dummy_optimizer.zero_grad()

    del dummy_optimizer
    torch.cuda.empty_cache()
    gc.collect()

    print("OOM preflight check complete")


def train_model(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    eval_datasets: dict[str, list[TrainingDataPoint]],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool = False,
):
    model = load_model(cfg.model_name, dtype)

    if cfg.gradient_checkpointing:
        model.use_cache = False
        model.gradient_checkpointing_enable()

    submodule = get_hf_submodule(model, cfg.hook_onto_layer)

    if cfg.use_lora and cfg.load_lora_path is None:
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif cfg.load_lora_path is not None:
        load_lora_path = Path(cfg.load_lora_path)
        assert load_lora_path.exists()
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True)
        model.print_trainable_parameters()

    model.train()

    oom_preflight_check(cfg, training_data, model, submodule, tokenizer, device, dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    total_training_steps = (cfg.num_epochs * len(training_data)) // cfg.train_batch_size
    # 10 percent
    warmup_steps = int(total_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    # --------------------------------------------------------------

    global_step = 0

    if os.path.exists("eval_logs.json"):
        os.remove("eval_logs.json")

    wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))

    for epoch in range(cfg.num_epochs):
        for i in tqdm(
            range(0, len(training_data), cfg.train_batch_size),
            desc=f"Training epoch {epoch + 1}",
        ):
            t_batch_list: list[TrainingDataPoint] = training_data[i : i + cfg.train_batch_size]

            t_batch = construct_batch(t_batch_list, tokenizer, device)

            if i % 10000 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            loss = train_features_batch(cfg, t_batch, model, submodule, device, dtype)
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                },
                step=global_step,
            )
            if verbose:
                print(f"Step {global_step} loss: {loss.item()}")

            # -------------------------------- evaluation --------------------------------
            if global_step % cfg.eval_steps == 0 and (cfg.eval_on_start or global_step > 0):
                for ds in eval_datasets:
                    eval_responses = run_evaluation(
                        cfg=cfg,
                        eval_data=eval_datasets[ds],
                        model=model,
                        tokenizer=tokenizer,
                        submodule=submodule,
                        device=device,
                        dtype=dtype,
                        global_step=global_step,
                    )
                    percent_format_correct, percent_ans_correct = score_eval_responses(
                        eval_responses, eval_datasets[ds]
                    )
                    wandb.log(
                        {
                            f"eval/{ds}_format_correct": percent_format_correct,
                            f"eval/{ds}_ans_correct": percent_ans_correct,
                        },
                        step=global_step,
                    )
                    print(
                        f"Step {global_step} {ds} format correct: {percent_format_correct}, ans correct: {percent_ans_correct}"
                    )
                model.train()

            if global_step % cfg.save_steps == 0 and global_step > 0:
                model.save_pretrained(f"{cfg.save_dir}/step_{global_step}")
                # Push to hF
                if cfg.hf_push_to_hub and cfg.hf_repo_id:
                    print("Pushing LoRA adapter to Hugging Face Hub...")
                    push_lora_to_hf(
                        model=model,
                        tokenizer=tokenizer,
                        repo_id=cfg.hf_repo_id + f"-step-{global_step}",
                        private=cfg.hf_private_repo,
                        commit_message=f"SAE introspection LoRA - {cfg.wandb_run_name} - step {global_step}",
                    )
                    print("Pushed LoRA adapter to Hugging Face Hub.")

            global_step += 1

    print("Training complete.")

    # Save final model
    print("Saving final model...")
    model.save_pretrained(f"{cfg.save_dir}/final")

    # Final evaluation
    print("Running final evaluation...")
    for ds in eval_datasets:
        eval_responses = run_evaluation(
            cfg=cfg,
            eval_data=eval_datasets[ds],
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            global_step=global_step,
        )
        percent_format_correct, percent_ans_correct = score_eval_responses(eval_responses, eval_datasets[ds])
        wandb.log(
            {
                f"eval/{ds}_format_correct": percent_format_correct,
                f"eval/{ds}_ans_correct": percent_ans_correct,
            },
            step=global_step,
        )
        print(f"Step {global_step} {ds} format correct: {percent_format_correct}, ans correct: {percent_ans_correct}")

    wandb.finish()

    # Push to Hugging Face if configured
    if cfg.hf_push_to_hub and cfg.hf_repo_id:
        print("Pushing LoRA adapter to Hugging Face Hub...")
        push_lora_to_hf(
            model=model,
            tokenizer=tokenizer,
            repo_id=cfg.hf_repo_id,
            commit_message=f"SAE introspection LoRA - {cfg.wandb_run_name} - final model",
            private=cfg.hf_private_repo,
        )


def length_grouped_reorder(
    data: list[TrainingDataPoint],
    batch_size: int,
    window_mult: int,
) -> list[TrainingDataPoint]:
    lengths = [len(d.input_ids) for d in data]

    indices = list(range(len(data)))
    megabatch_size = window_mult * batch_size

    # Slice into mega-batches
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(indices), megabatch_size)]
    # Sort within each mega-batch by length desc
    megabatches = [sorted(mb, key=lambda i: lengths[i], reverse=True) for mb in megabatches]

    new_order = [i for mb in megabatches for i in mb]
    return [data[i] for i in new_order]


def build_datasets(
    cfg: SelfInterpTrainingConfig,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    max_len_percentile: float | None = 0.999,
    window_mult: int | None = 20,
) -> tuple[list[TrainingDataPoint], list[TrainingDataPoint]]:
    random.seed(cfg.seed)
    all_training_data: list[TrainingDataPoint] = []
    # eval data will only be for classification datasets
    all_eval_data: dict[str, list[TrainingDataPoint]] = {}

    for train_dataset_loader in cfg.train_dataset_loaders:
        all_training_data.extend(train_dataset_loader.load_dataset("train"))
    for eval_dataset_loader in cfg.eval_dataset_loaders:
        all_eval_data[eval_dataset_loader.dataset_params.classification_dataset_name] = (
            eval_dataset_loader.load_dataset("test")
        )

    p = max_len_percentile
    if p is not None:
        if p >= 1.0 or p <= 0.0:
            raise ValueError("max_len_percentile must be less than 1.0 and greater than 0.0")

        lengths = sorted(len(td.input_ids) for td in all_training_data)
        median_length = lengths[len(lengths) // 2]
        print(f"Max length: {lengths[-1]}, Min length: {lengths[0]}, Median length: {median_length}")
        # Inclusive quantile index
        idx = int((len(lengths) - 1) * p)
        threshold = lengths[idx]

        before = len(all_training_data)
        all_training_data = [td for td in all_training_data if len(td.input_ids) <= threshold]
        removed = before - len(all_training_data)
        print(f"Percentile trim: kept <= {threshold} tokens (p={p:.6f}). Removed {removed}/{before} examples.")

    random.seed(cfg.seed)
    random.shuffle(all_training_data)

    if window_mult is not None:
        all_training_data = length_grouped_reorder(all_training_data, cfg.train_batch_size, window_mult)

    return all_training_data, all_eval_data


if __name__ == "__main__":
    main_train_size = 6000
    classification_datasets_train_sizes = {
        "geometry_of_truth": main_train_size,
        "relations": main_train_size,
        "sst2": main_train_size,
        "md_gender": main_train_size,
        "snli": main_train_size,
        "ag_news": main_train_size,
        "ner": main_train_size,
        "tense": main_train_size,
        "language_identification": main_train_size,
        "singular_plural": 10,  # very small dataset
    }
    classification_eval_datasets = [
        "geometry_of_truth",
        "relations",
        "sst2",
        "md_gender",
        "snli",
        "ag_news",
        "ner",
        "tense",
        "language_identification",
        "singular_plural",
    ]
    classification_train_datasets = [
        "geometry_of_truth",
        "relations",
        "sst2",
        "md_gender",
        "snli",
        # "ag_news",
        "ner",
        "tense",
        # "language_identification",
        # "singular_plural",
    ]

    hook_layer = 1
    model_name = "Qwen/Qwen3-8B"
    hf_repo_name = f"qwen3-8b-hook-layer-{hook_layer}"

    device = torch.device("cuda")
    dtype = torch.bfloat16

    layer_percents = [25, 50, 75]

    train_dataset_loaders = []
    eval_dataset_loaders = []
    sft_data_folder = "sft_training_data"
    seed = 42

    dataset_config = DatasetLoaderConfig(
        custom_dataset_params=PastLensDatasetConfig(),
        num_train=300,
        num_test=0,
        splits=["train"],
        model_name=model_name,
        layer_percents=layer_percents,
        save_acts=True,
    )

    past_lens_dataset_loader = PastLensDatasetLoader(
        dataset_config=dataset_config,
    )

    train_dataset_loaders.append(past_lens_dataset_loader)

    for dataset_name in classification_datasets_train_sizes.keys():
        classification_config = ClassificationDatasetConfig(
            classification_dataset_name=dataset_name,
        )

        dataset_config = DatasetLoaderConfig(
            custom_dataset_params=classification_config,
            num_train=classification_datasets_train_sizes[dataset_name],
            num_test=250,
            splits=["train", "test"],
            model_name=model_name,
            layer_percents=layer_percents,
            save_acts=True,
        )

        classification_dataset_loader = ClassificationDatasetLoader(
            dataset_config=dataset_config,
        )

        if dataset_name in classification_train_datasets:
            train_dataset_loaders.append(classification_dataset_loader)

        if dataset_name in classification_eval_datasets:
            eval_dataset_loaders.append(classification_dataset_loader)

    iterations = [
        {
            "load_lora_path": None,
            "train_dataset_loaders": train_dataset_loaders,
            "eval_dataset_loaders": eval_dataset_loaders,
            "wandb_suffix": "_act_pretrain",
        },
    ]

    for hyperparam_override in iterations:
        cfg = SelfInterpTrainingConfig(
            model_name=model_name,
            hook_onto_layer=hook_layer,
            hf_repo_name=hf_repo_name,
            # wandb_suffix=wandb_suffix,
            layer_percents=layer_percents,
            train_batch_size=16,
            activation_collection_batch_size=64,
            eval_steps=1000,
            eval_on_start=False,
            **hyperparam_override,
        )

        cfg.finalize()

        print(f"save dir: {cfg.save_dir}")

        tokenizer = load_tokenizer(cfg.model_name)

        all_training_data, all_eval_data = build_datasets(cfg, tokenizer, device, dtype, window_mult=cfg.window_mult)

        # for debugging
        # all_training_data = all_training_data[:1000]

        print(f"training data: {len(all_training_data)}, eval data: {len(all_eval_data)}")

        print(asdict(cfg))

        train_model(
            cfg=cfg,
            training_data=all_training_data,
            eval_datasets=all_eval_data,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            verbose=True,
        )
