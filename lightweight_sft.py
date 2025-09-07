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

from detection_eval.steering_hooks import add_hook, get_hf_activation_steering_hook, get_introspection_prompt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import contextlib
import datetime
import gc
import json
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

# ==============================================================================
# 1. HUGGING FACE SETUP
# ==============================================================================


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


# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================


@dataclass
class SelfInterpTrainingConfig:
    # --- Model ---
    model_name: str = "Qwen/Qwen3-8B"
    hook_onto_layer: int = 1
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])
    act_layers: list[int] = field(default_factory=list)  # derived if empty

    # --- Data / experiment ---
    sae_sft_datasets: list[str] = field(default_factory=list)  # pass in or compute outside
    classification_datasets: list[str] = field(default_factory=lambda: ["sst2", "ag_news"])
    use_decoder_vectors: bool = True
    generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"do_sample": True, "temperature": 1.0, "max_new_tokens": 300}
    )
    steering_coefficient: float = 2.0
    act_collect_offset: int = -4
    max_sae_sft_examples: int = 50_000
    max_classification_examples: int = 10_000
    test_set_size_per_ds: int = 25
    dataset_folder: str = "sft_training_data"
    num_qa_per_sample: int = 3

    # --- Batching ---
    train_batch_size: int = 4
    eval_batch_size: int = 128
    activation_collection_batch_size: int = 128

    # --- LoRA ---
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # --- Training ---
    num_epochs: int = 1
    lr: float = 2e-5
    max_grad_norm: float = 1.0
    eval_steps: int = 9_999_999  # effectively off by default
    save_steps: int = 2_000
    save_dir: str = "checkpoints"
    seed: int = 42
    eval_logs_path: str = "eval_logs.json"

    # --- Tracking ---
    wandb_project: str = "sae_introspection"
    wandb_run_name: str = ""  # derived if empty
    wandb_suffix: str = ""

    # --- Hub ---
    hf_push_to_hub: bool = False
    hf_private_repo: bool = False
    hf_repo_name: str = ""  # optional short name, used to compute repo_id
    hf_repo_id: str = ""  # derived if empty and push is on

    # --- Misc experiment options ---
    positive_negative_examples: bool = False

    def finalize(self) -> "SelfInterpTrainingConfig":
        # act_layers from percents if caller did not set them directly
        if not self.act_layers:
            self.act_layers = [layer_percent_to_layer(self.model_name, p) for p in self.layer_percents]

        # run name - stable and readable
        layers_str = "-".join(map(str, self.act_layers))
        default_run = f"{self.model_name}-layers_{layers_str}-decoder-{self.use_decoder_vectors}{self.wandb_suffix}"
        if not self.wandb_run_name:
            self.wandb_run_name = default_run

        # save dir namespacing
        if self.wandb_suffix and not self.save_dir.endswith(self.wandb_suffix):
            self.save_dir = f"{self.save_dir}{self.wandb_suffix}"

        # repo id if pushing
        if self.hf_push_to_hub and not self.hf_repo_id:
            self.hf_repo_id = get_hf_repo_id(self.hf_repo_name)
        return self


# ==============================================================================
# 3. DATA MODELS
# ==============================================================================


class SAEExplained(BaseModel):
    sae_id: int
    sae_info: SAEInfo
    explanation: str
    positive_examples: list[str]
    negative_examples: list[str]
    f1: float


class ExplanationResult(BaseModel):
    """Parsed explanation from model generation."""

    explanation: str


class TrainingExample(BaseModel):
    """Training example with explanation and metadata."""

    explanation: str
    feature_idx: int

    @classmethod
    def with_positive_and_negative_examples(cls, sae_explanation: SAEExplained) -> "TrainingExample":
        positive_examples_text = "".join(
            f"<positive_example>{example}</positive_example>\n" for example in sae_explanation.positive_examples
        )

        negative_examples_text = "".join(
            f"<negative_example>{example}</negative_example>\n" for example in sae_explanation.negative_examples
        )

        prompt = f"""{positive_examples_text.rstrip()}
{negative_examples_text.rstrip()}
<explanation>{sae_explanation.explanation}</explanation>"""

        return TrainingExample(
            explanation=prompt,
            feature_idx=sae_explanation.sae_id,
        )

    @classmethod
    def with_explanation_only(cls, sae_explanation: SAEExplained) -> "TrainingExample":
        prompt = f"<explanation>{sae_explanation.explanation}</explanation>"
        return TrainingExample(
            explanation=prompt,
            feature_idx=sae_explanation.sae_id,
        )


class SentenceData(BaseModel):
    """Data about a sentence pair."""

    original_sentence: str
    rewritten_sentence: str


class SentenceMetrics(BaseModel):
    """Metrics for sentence evaluation."""

    original_max_activation: float
    rewritten_max_activation: float
    sentence_distance: float


class FeatureResult(BaseModel):
    """Result for a single feature evaluation."""

    feature_idx: int
    api_response: str
    prompt: str
    explanation: str


class EvalStepResult(BaseModel):
    """Results from a single evaluation step."""

    step: int
    results: list[FeatureResult]


class TrainingDataPoint(BaseModel):
    """Training data point with tensors."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_ids: list[int]
    labels: list[int]  # Can contain -100 for ignored tokens
    steering_vectors: list[torch.Tensor]
    positions: list[int]
    feature_idx: int
    target_output: str

    @field_validator("positions")
    @classmethod
    def _len_match(cls, pos, info):
        # Ensure positions and steering_vectors align
        sv = info.data.get("steering_vectors", [])
        if sv and len(pos) != len(sv):
            raise ValueError("positions and steering_vectors must have the same length")
        return pos


class BatchData(BaseModel):
    """Batch of training data with tensors."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[int]
    feature_indices: list[int]


# ==============================================================================
# 4. MODEL UTILITIES
# ==============================================================================


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    """Convert a layer percent to a layer number."""
    if model_name == "Qwen/Qwen3-8B":
        max_layers = 36
        return int(max_layers * (layer_percent / 100))
    else:
        raise ValueError(f"Unknown model name: {model_name}")


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


def load_explanations_from_jsonl(filepath: str) -> list[SAEExplained]:
    """Load SAE explanations from a JSONL file."""
    explanations = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                explanations.append(SAEExplained(**data))
    return explanations


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
    input_messages = [{"role": "user", "content": input_prompt}]

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

    training_data = []

    for i in tqdm(range(dataset_size), desc="Constructing training dataset"):
        target_response = training_examples[i].explanation

        full_messages = input_messages + [{"role": "assistant", "content": target_response}]

        if i == 0:
            # Fully print the first example
            print("First training example:")
            print(full_messages)
            print("-" * 100)

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
        target_feature_idx = training_examples[i].feature_idx

        # 2. Prepare feature vectors for steering
        # We use decoder weights (W_dec) as they map from the feature space back to the residual stream.
        # .clone() because otherwise we will save the entire W_dec in pickle for each training example
        if cfg.use_decoder_vectors:
            feature_vector = sae.W_dec[target_feature_idx].clone()
        else:
            feature_vector = sae.W_enc[:, target_feature_idx].clone()

        assistant_start_idx = len(input_prompt_ids)

        labels = full_prompt_ids.copy()
        for i in range(assistant_start_idx):
            labels[i] = -100

        positions = []
        for i in range(assistant_start_idx):
            if full_prompt_ids[i] == x_token_id:
                positions.append(i)
        assert len(positions) == 1, "Expected exactly one X token"

        training_data_point = TrainingDataPoint(
            input_ids=full_prompt_ids,
            labels=labels,
            steering_vectors=[feature_vector],
            positions=positions,
            feature_idx=target_feature_idx,
            target_output=target_response,
        )

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


def construct_batch(
    training_data: list[TrainingDataPoint],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> BatchData:
    max_length = 0
    for data_point in training_data:
        max_length = max(max_length, len(data_point.input_ids))

    batch_tokens = []
    batch_labels = []
    batch_attn_masks = []
    batch_positions = []
    batch_steering_vectors = []
    batch_feature_indices = []

    for data_point in training_data:
        padding_length = max_length - len(data_point.input_ids)
        padding_tokens = [tokenizer.pad_token_id] * padding_length
        padded_input_ids = padding_tokens + data_point.input_ids
        padded_labels = [-100] * padding_length + data_point.labels

        input_ids = torch.tensor(padded_input_ids, dtype=torch.long).to(device)
        labels = torch.tensor(padded_labels, dtype=torch.long).to(device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool).to(device)

        attn_mask[:padding_length] = False

        batch_tokens.append(input_ids)
        batch_labels.append(labels)
        batch_attn_masks.append(attn_mask)

        # Extract single position and single steering vector (simplified structure)
        assert len(data_point.positions) == 1, f"Expected exactly one position, got {len(data_point.positions)}"
        assert len(data_point.steering_vectors) == 1, (
            f"Expected exactly one steering vector, got {len(data_point.steering_vectors)}"
        )

        single_position = data_point.positions[0] + padding_length
        single_steering_vector = data_point.steering_vectors[0].to(device)

        batch_positions.append(single_position)
        batch_steering_vectors.append(single_steering_vector)
        batch_feature_indices.append(data_point.feature_idx)

    return BatchData(
        input_ids=torch.stack(batch_tokens),
        labels=torch.stack(batch_labels),
        attention_mask=torch.stack(batch_attn_masks),
        steering_vectors=batch_steering_vectors,
        positions=batch_positions,
        feature_indices=batch_feature_indices,
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

    # Generate both samples first, then display them grouped by feature
    all_samples = []
    all_explanations = []

    for sample_idx in range(2):
        with add_hook(submodule, hook_fn):
            output_ids = model.generate(**tokenized_input, **cfg.generation_kwargs)

        # Decode only the newly generated tokens
        generated_tokens = output_ids[:, eval_batch.input_ids.shape[1] :]
        decoded_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        explanations = []
        for output in decoded_output:
            explanations.append(parse_generated_explanation(output))

        all_samples.append(decoded_output)
        all_explanations.append(explanations)

    # Now display and process both samples for each feature consecutively
    for i in range(len(eval_batch.feature_indices)):
        feature_idx = eval_batch.feature_indices[i]

        print(f"\n=== Feature {feature_idx} ===")

        # Show both samples for this feature
        for sample_idx in range(2):
            output = all_samples[sample_idx][i]
            print(f"Sample {sample_idx + 1}: {output}")

            # Extract explanation string, handling None case
            explanation_str = ""
            if all_explanations[sample_idx][i] is not None:
                explanation_str = all_explanations[sample_idx][i].explanation

            feature_result = FeatureResult(
                feature_idx=feature_idx,
                api_response=output,
                prompt=decoded_prompts[i],
                explanation=explanation_str,
            )
            feature_results.append(feature_result)

        print()  # Empty line for readability

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

        save_logs(
            eval_results_path="eval_logs.json",
            global_step=global_step,
            all_feature_results_this_eval_step=all_feature_results,
        )
    return all_feature_results


def train_model(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    eval_data: list[TrainingDataPoint],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool = False,
    load_lora_path: Optional[Path] = None,
):
    model = load_model(cfg.model_name, dtype)
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

            if i % 100 == 0:
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
            if global_step % cfg.eval_steps == 0 and global_step > 0:
                run_evaluation(
                    cfg=cfg,
                    eval_data=eval_data,
                    model=model,
                    tokenizer=tokenizer,
                    submodule=submodule,
                    device=device,
                    dtype=dtype,
                    global_step=global_step,
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

    wandb.finish()

    # Final evaluation
    print("Running final evaluation...")
    run_evaluation(
        cfg=cfg,
        eval_data=eval_data,
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        device=device,
        dtype=dtype,
        global_step=global_step,
    )

    # Save final model
    print("Saving final model...")
    model.save_pretrained(f"{cfg.save_dir}/final")

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


def get_hf_repo_id(hf_repo_name: str) -> str:
    print("Setting up Hugging Face authentication...")
    # check if already logged in
    if whoami() is None:
        print("Not logged in to Hugging Face. Attempting to log in...")
        login()
    else:
        print("Already logged in to Hugging Face.")

    # Determine default HF repo name if not provided
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    if not hf_repo_name:
        hf_repo_name = f"gemma-introspection-{date_str}"

    # Compose full repo_id with current username
    user_info = whoami()
    owner = user_info.get("name") if isinstance(user_info, dict) else None
    hf_repo_id_computed = f"{owner}/{hf_repo_name}" if owner else hf_repo_name

    return hf_repo_id_computed


def build_datasets(
    cfg: SelfInterpTrainingConfig,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[TrainingDataPoint], list[TrainingDataPoint], list[SAEInfo]]:
    all_training_data: list[TrainingDataPoint] = []
    all_eval_data: list[TrainingDataPoint] = []
    all_sae_infos: list[SAEInfo] = []

    # SFT-style feature explanations
    for sft_file in cfg.sae_sft_datasets:
        file_data, sae_info = load_sae_data_from_sft_data_file(sft_file, cfg, tokenizer, device, dtype)
        file_data = file_data[: cfg.max_sae_sft_examples]
        all_training_data.extend(file_data[: -cfg.test_set_size_per_ds])
        all_eval_data.extend(file_data[-cfg.test_set_size_per_ds :])
        all_sae_infos.append(sae_info)

    # Classification side-task
    for ds in cfg.classification_datasets:
        print(f"Creating classification dataset for {ds}")
        train_ds, test_ds = classification.create_classification_dataset(
            ds,
            num_qa_per_sample=cfg.num_qa_per_sample,
            num_train_examples=cfg.max_classification_examples,
            num_test_examples=cfg.test_set_size_per_ds,
            batch_size=cfg.activation_collection_batch_size,
            act_layers=cfg.act_layers,
            offset=cfg.act_collect_offset,
            model_name=cfg.model_name,
            tokenizer=tokenizer,
            dtype=dtype,
            random_seed=cfg.seed,
            dataset_folder=cfg.dataset_folder,
        )
        all_training_data.extend(train_ds)
        all_eval_data.extend(test_ds)

    random.seed(cfg.seed)
    random.shuffle(all_training_data)
    random.shuffle(all_eval_data)

    return all_training_data, all_eval_data, all_sae_infos


if __name__ == "__main__":
    classification_datasets = ["sst2", "ag_news"]

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

    for use_decoder_vectors in [True]:
        wandb_suffix = f"_larger_dataset_layer_{hook_layer}"
        if use_decoder_vectors:
            wandb_suffix += "_decoder"
        else:
            wandb_suffix += "_encoder"

        cfg = SelfInterpTrainingConfig(
            model_name=model_name,
            hook_onto_layer=hook_layer,
            hf_repo_name=hf_repo_name,
            wandb_suffix=wandb_suffix,
            layer_percents=layer_percents,
            sae_sft_datasets=explanations_files,
            classification_datasets=classification_datasets,
            max_classification_examples=10_000,
        )

        # mutate the cfg here using variables in the itertools loop over variables of interest
        cfg.use_decoder_vectors = use_decoder_vectors

        cfg.finalize()

        tokenizer = load_tokenizer(cfg.model_name)

        all_training_data, all_eval_data, all_sae_infos = build_datasets(cfg, tokenizer, device, dtype)

        # for debugging
        all_training_data = all_training_data[:1000]

        print(f"training data: {len(all_training_data)}, eval data: {len(all_eval_data)}")

        cfg.sae_infos = all_sae_infos

        print(asdict(cfg))

        train_model(
            cfg=cfg,
            training_data=all_training_data,
            eval_data=all_eval_data,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            verbose=True,
        )
