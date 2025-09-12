"""
Lightweight SAE Introspection Training Script

This script trains a model to understand SAE (Sparse Autoencoder) features through introspection.

Features:
- Automatic Hugging Face login at script start
- LoRA fine-tuning support
- Automatic pushing of trained LoRA adapters to Hugging Face Hub after training
- Configurable repository settings (public/private)

Usage:
    python lightweight_sft.py [explanations_file.jsonl]

Before running:
1. Make sure you're logged into Hugging Face: `huggingface-cli login`
2. Update the hf_repo_id in the main() function to your desired repository name
3. Ensure you have the required explanations JSONL file
"""

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

import classification
import wandb
from create_hard_negatives_v2 import (
    BaseSAE,
    JumpReluSAE,
    get_sae_info,
    get_submodule,
    load_model,
    load_sae,
    load_tokenizer,
)
from detection_eval.detection_basemodels import SAEInfo
from detection_eval.steering_hooks import add_hook, get_hf_activation_steering_hook, get_introspection_prompt
from sft_config import (
    BatchData,
    EvalStepResult,
    ExplanationResult,
    FeatureResult,
    SAEExplained,
    SelfInterpTrainingConfig,
    TrainingDataPoint,
    TrainingExample,
    construct_batch,
    create_training_datapoint,
    load_explanations_from_jsonl,
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


class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


def collect_activations(
    model: AutoModelForCausalLM,
    submodule: torch.nn.Module,
    inputs_BL: dict[str, torch.Tensor],
    use_no_grad: bool = True,
) -> torch.Tensor:
    """
    Collects activations from a specific submodule (layer) during model forward pass.

    Args:
        model: The transformer model
        submodule: The specific layer/module to collect activations from
        inputs_BL: Tokenized inputs (batch, length)
        use_no_grad: Whether to use torch.no_grad() for efficiency

    Returns:
        activations_BLD: Activations tensor (batch, length, hidden_dim)
    """
    activations_BLD = None

    def hook(module, input, output):
        nonlocal activations_BLD
        if isinstance(output, tuple):
            activations_BLD = output[0]  # For models that return tuples
        else:
            activations_BLD = output
        # Stop computation early if we only need activations
        raise EarlyStopException()

    # Register hook
    handle = submodule.register_forward_hook(hook)

    try:
        ctx = torch.no_grad() if use_no_grad else contextlib.nullcontext()
        with ctx:
            try:
                model(**inputs_BL)
            except EarlyStopException:
                pass  # Expected - we stopped early to collect activations
    finally:
        handle.remove()

    if activations_BLD is None:
        raise RuntimeError("Failed to collect activations")

    return activations_BLD


# ==============================================================================
# 6. UTILITY FUNCTIONS
# ==============================================================================


def build_training_prompt(positive_negative_examples: bool, sae_layer: int) -> str:
    """Build the training prompt for SAE explanations."""
    if positive_negative_examples:
        raise NotImplementedError("Not implemented")
        question = f"""Can you explain to me the concept of what 'X' from layer {sae_layer} means? Give positive and negative examples of what the concept would activate on. Format your final answer with <explanation>."""
    else:
        question = get_introspection_prompt(sae_layer)
    return question


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


# ==============================================================================
# 3. HOOKING MECHANISM FOR ACTIVATION STEERING
# ==============================================================================


@torch.no_grad()
def construct_train_dataset(
    cfg: SelfInterpTrainingConfig,
    dataset_size: int,
    input_prompt: str,
    training_examples: list[TrainingExample],
    sae: BaseSAE,
    tokenizer: PreTrainedTokenizer,
) -> list[TrainingDataPoint]:
    training_data = []

    for i in tqdm(range(dataset_size), desc="Constructing training dataset"):
        target_response = training_examples[i].explanation
        target_feature_idx = training_examples[i].feature_idx
        # 2. Prepare feature vectors for steering
        # We use decoder weights (W_dec) as they map from the feature space back to the residual stream.
        # .clone() because otherwise we will save the entire W_dec in pickle for each training example
        if cfg.use_decoder_vectors:
            feature_vector = sae.W_dec[target_feature_idx].clone()
        else:
            feature_vector = sae.W_enc[:, target_feature_idx].clone()

        training_data_point = create_training_datapoint(
            prompt=input_prompt,
            target_response=target_response,
            tokenizer=tokenizer,
            acts_D=feature_vector,
            feature_idx=target_feature_idx,
        )

        if i == 0:
            # Fully print the first example
            print("First training example:")
            print(f"prompt: {input_prompt}")
            print(f"target_response: {target_response}")
            print("-" * 100)

        training_data.append(training_data_point)

    return training_data


def construct_eval_dataset(
    cfg: SelfInterpTrainingConfig,
    dataset_size: int,
    input_prompt: str,
    eval_feature_indices: list[int],
    api_data: dict,
    sae: BaseSAE,
    tokenizer: PreTrainedTokenizer,
    enable_thinking: bool = False,
) -> list[TrainingDataPoint]:
    """Every prompt is exactly the same - the only difference is the steering vectors."""

    input_messages = [{"role": "user", "content": input_prompt}]

    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=enable_thinking,
    )
    if not isinstance(input_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")
    labels = input_prompt_ids.copy()

    orig_prompt_length = len(input_prompt_ids)

    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]

    eval_data = []

    first_position = None

    for i in tqdm(range(dataset_size), desc="Constructing eval dataset"):
        target_feature_idx = eval_feature_indices[i]

        # 2. Prepare feature vectors for steering
        # We use decoder weights (W_dec) as they map from the feature space back to the residual stream.
        if cfg.use_decoder_vectors:
            feature_vector = sae.W_dec[target_feature_idx].clone()
        else:
            feature_vector = sae.W_enc[:, target_feature_idx].clone()

        positions = []
        for i in range(orig_prompt_length):
            if input_prompt_ids[i] == x_token_id:
                positions.append(i)

        assert len(positions) == 1, "Expected exactly one X token"

        if first_position is None:
            first_position = positions[0]
        else:
            assert positions[0] == first_position, "Expected all positions to be the same"
        assert len(input_prompt_ids) > 0

        eval_data_point = TrainingDataPoint(
            input_ids=input_prompt_ids,
            labels=labels,
            steering_vectors=[feature_vector],
            positions=positions,
            feature_idx=target_feature_idx,
            target_output="",
        )

        eval_data.append(eval_data_point)

    return eval_data


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


# ==============================================================================


# ==============================================================================
# 8. INTROSPECTION UTILITIES
# ==============================================================================


def get_bos_eos_pad_mask(tokenizer: PreTrainedTokenizer, token_ids: torch.Tensor) -> torch.Tensor:
    """Create mask for BOS, EOS, and PAD tokens"""
    mask = torch.zeros_like(token_ids, dtype=torch.bool)

    if tokenizer.bos_token_id is not None:
        mask |= token_ids == tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        mask |= token_ids == tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        mask |= token_ids == tokenizer.pad_token_id

    return mask


def get_feature_activations(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    sae: JumpReluSAE,
    tokenized_strs: dict[str, torch.Tensor],
    ignore_bos: bool = True,
) -> torch.Tensor:
    with torch.no_grad():
        pos_acts_BLD = collect_activations(model, submodule, tokenized_strs)
        encoded_pos_acts_BLF = sae.encode(pos_acts_BLD)

    if ignore_bos:
        bos_mask = tokenized_strs["input_ids"] == tokenizer.bos_token_id
        # Note: I use >=, not ==, because occasionally prompts will contain a BOS token
        assert bos_mask.sum() >= encoded_pos_acts_BLF.shape[0], (
            f"Expected at least {encoded_pos_acts_BLF.shape[0]} BOS tokens, but found {bos_mask.sum()}"
        )

        mask = get_bos_eos_pad_mask(tokenizer, tokenized_strs["input_ids"])
        encoded_pos_acts_BLF[mask] = 0

    return encoded_pos_acts_BLF


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
    load_lora_path: Optional[Path] = None,
):
    model = load_model(cfg.model_name, dtype)

    if cfg.gradient_checkpointing:
        model.use_cache = False
        model.gradient_checkpointing_enable()

    submodule = get_submodule(model, cfg.hook_onto_layer)

    if cfg.use_lora and load_lora_path is None:
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
    elif load_lora_path is not None:
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


def load_sae_data_from_sft_data_file(
    sft_data_file: str,
    cfg: SelfInterpTrainingConfig,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[TrainingDataPoint], SAEInfo]:
    explanations: list[SAEExplained] = load_explanations_from_jsonl(sft_data_file)
    orig_sae_info = explanations[0].sae_info
    for data_point in explanations:
        assert data_point.sae_info == orig_sae_info
    sae_info = SAEInfo.model_validate(orig_sae_info)

    sae = load_sae(sae_info.sae_repo_id, sae_info.sae_filename, sae_info.sae_layer, cfg.model_name, device, dtype)

    training_examples = [
        TrainingExample.with_positive_and_negative_examples(exp)
        if cfg.positive_negative_examples
        else TrainingExample.with_explanation_only(exp)
        for exp in explanations
    ]
    print(f"Loaded {len(training_examples)} training examples from {sft_data_file}")

    train_features = set()

    for example in training_examples:
        train_features.add(example.feature_idx)

    # For evaluation, we'll use a subset of the training features
    # In a real scenario, you might want to load a separate eval set
    print(f"train examples: {len(training_examples)}")
    print(f"Train features: {len(train_features)}")

    train_eval_prompt = build_training_prompt(cfg.positive_negative_examples, sae_info.sae_layer)

    training_data: list[TrainingDataPoint] = construct_train_dataset(
        cfg,
        len(training_examples),
        # dataset_size,
        train_eval_prompt,
        training_examples,
        sae,
        tokenizer,
    )

    return training_data, sae_info


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
) -> tuple[list[TrainingDataPoint], list[TrainingDataPoint], list[SAEInfo]]:
    all_training_data: list[TrainingDataPoint] = []

    # eval data will only be for classification datasets
    all_eval_data: dict[str, list[TrainingDataPoint]] = {}
    all_sae_infos: list[SAEInfo] = []

    # SFT-style feature explanations
    for sft_file in cfg.sae_sft_datasets:
        file_data, sae_info = load_sae_data_from_sft_data_file(sft_file, cfg, tokenizer, device, dtype)
        file_data = file_data[: cfg.max_sae_sft_examples]
        all_training_data.extend(file_data[: -cfg.test_set_size_per_ds])
        all_sae_infos.append(sae_info)

    for sft_file in cfg.additional_train_dataset_filenames:
        with open(sft_file, "rb") as f:
            file_data = pickle.load(f)
        all_training_data.extend(file_data)

    random.seed(cfg.seed)

    # Classification side-task
    for ds in cfg.classification_train_datasets:
        print(f"Creating train classification dataset for {ds}")
        train_ds, test_ds = classification.create_classification_dataset(
            ds,
            num_qa_per_sample=cfg.num_qa_per_sample,
            num_train_examples=cfg.max_classification_examples,
            num_test_examples=cfg.test_set_size_per_ds,
            batch_size=cfg.activation_collection_batch_size,
            act_layers=cfg.act_layers,
            min_offset=cfg.min_act_collect_offset,
            max_offset=cfg.max_act_collect_offset,
            model_name=cfg.model_name,
            tokenizer=tokenizer,
            dtype=dtype,
            random_seed=cfg.seed,
            dataset_folder=cfg.dataset_folder,
        )
        all_training_data.extend(train_ds)

    for ds in cfg.classification_eval_datasets:
        print(f"Creating test classification dataset for {ds}")
        test_ds = classification.create_classification_dataset_test_only(
            ds,
            num_qa_per_sample=cfg.num_qa_per_sample,
            num_test_examples=cfg.test_set_size_per_ds,
            batch_size=cfg.activation_collection_batch_size,
            act_layers=cfg.act_layers,
            min_offset=cfg.min_act_collect_offset,
            max_offset=cfg.max_act_collect_offset,
            model_name=cfg.model_name,
            tokenizer=tokenizer,
            dtype=dtype,
            random_seed=cfg.seed,
            dataset_folder=cfg.dataset_folder,
        )
        all_eval_data[ds] = test_ds

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

    return all_training_data, all_eval_data, all_sae_infos


if __name__ == "__main__":
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

    explanations_files = []
    for layer_percent in layer_percents:
        explanations_files.append(
            f"data/qwen_hard_negatives_0_20000_layer_percent_{layer_percent}_sft_data_gpt-5-mini-2025-08-07.jsonl"
        )

    additional_explanations_files = []

    for layer_percent in layer_percents:
        act_examples_filename = (
            f"sft_training_data/act_examples_Qwen_Qwen3-8B_layer_percent_{layer_percent}_width_2_num_features_60000.pkl"
        )
        sae_yes_no_filename = f"sft_training_data/yes_no_sae_data_Qwen_Qwen3-8B_layer_percent_{layer_percent}_width_2_max_features_None.pkl"
        additional_explanations_files.append(act_examples_filename)
        additional_explanations_files.append(sae_yes_no_filename)

    # explanations_files = []

    load_lora_path = Path("checkpoints_sae_layer_1_decoder/final")

    iterations = [
        # {"lr": 2e-5},
        # {"lr": 5e-5},
        # {"act_collect_offset": -3},
        # {"act_collect_offset": -5},
        # {"num_epochs": 2},
        {"min_act_collect_offset": -2, "max_act_collect_offset": -5},
        # {}
    ]

    # for use_decoder_vectors in [True]:
    for hyperparam_override in iterations:
        wandb_suffix = f"_no_sae_multiple_datasets_layer_{hook_layer}_larger_pretrain"
        # wandb_suffix = f"_with_sae_multiple_datasets_layer_{hook_layer}_window_mult_{window_mult}"
        hyperparam_suffix = ""
        for key, value in hyperparam_override.items():
            hyperparam_suffix += f"_{key}_{value}"
        wandb_suffix += hyperparam_suffix
        print(wandb_suffix)
        # wandb_suffix = f"_sae_layer_{hook_layer}"
        # if use_decoder_vectors:
        #     wandb_suffix += "_decoder"
        # else:
        #     wandb_suffix += "_encoder"

        cfg = SelfInterpTrainingConfig(
            model_name=model_name,
            hook_onto_layer=hook_layer,
            hf_repo_name=hf_repo_name,
            wandb_suffix=wandb_suffix,
            layer_percents=layer_percents,
            sae_sft_datasets=explanations_files,
            classification_train_datasets=classification_train_datasets,
            classification_eval_datasets=classification_eval_datasets,
            additional_train_dataset_filenames=additional_explanations_files,
            max_classification_examples=6_000,
            test_set_size_per_ds=250,
            train_batch_size=16,
            activation_collection_batch_size=64,
            eval_steps=10000,
            eval_on_start=False,
            load_lora_path=str(load_lora_path),
            # act_collect_offset=-4,
            **hyperparam_override,
        )

        # mutate the cfg here using variables in the itertools loop over variables of interest
        # cfg.use_decoder_vectors = use_decoder_vectors
        # cfg.act_collect_offset = offset

        cfg.finalize()

        tokenizer = load_tokenizer(cfg.model_name)

        all_training_data, all_eval_data, all_sae_infos = build_datasets(
            cfg, tokenizer, device, dtype, window_mult=cfg.window_mult
        )

        # for debugging
        # all_training_data = all_training_data[:1000]

        print(f"training data: {len(all_training_data)}, eval data: {len(all_eval_data)}")

        # raise Exception("Stop here")

        cfg.sae_infos = all_sae_infos

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
