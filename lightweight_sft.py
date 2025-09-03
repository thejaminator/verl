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

from detection_eval.steering_hooks import X_PROMPT, add_hook, get_hf_activation_steering_hook

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import contextlib
import datetime
import gc
import json

# All necessary imports are now included above
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch
import wandb
from huggingface_hub import login, whoami
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer

from create_hard_negatives_v2 import BaseSAE, JumpReluSAE, get_sae_info, get_submodule, load_sae

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
    """Configuration settings for the script."""

    # --- Model Settings ---
    model_name: str
    train_batch_size: int
    eval_batch_size: int

    # --- SAE (Sparse Autoencoder) Settings ---
    sae_repo_id: str
    hook_onto_layer: int
    sae_layer: int
    sae_width: int

    # --- Experiment Settings ---
    eval_set_size: int
    use_decoder_vectors: bool
    generation_kwargs: dict[str, Any]
    steering_coefficient: float

    # --- LoRA Settings ---
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str

    # --- Training Settings ---
    num_epochs: int
    lr: float
    eval_steps: int
    save_steps: int
    save_dir: str

    # --- Hugging Face Settings ---
    hf_push_to_hub: bool
    hf_private_repo: bool
    hf_repo_id: str = "thejaminator/sae-introspection-lora"

    # --- Fields with defaults (must come after fields without defaults) ---
    sae_filename: str = field(init=False)
    eval_features: list[int] = field(default_factory=list)
    positive_negative_examples: bool = True

    def __post_init__(self):
        """Called after the dataclass is initialized."""
        info = get_sae_info(
            sae_repo_id=self.sae_repo_id, sae_width=None,
        )
        self.sae_filename = info.sae_filename


# ==============================================================================
# 3. DATA MODELS
# ==============================================================================


class SAEExplained(BaseModel):
    sae_id: int
    explanation: str
    positive_examples: list[str]
    negative_examples: list[str]


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


@dataclass
class TrainingDataPoint:
    """Training data point with tensors."""

    input_ids: list[int]
    labels: list[int]  # Can contain -100 for ignored tokens
    steering_vectors: list[torch.Tensor]
    positions: list[int]
    feature_idx: int


@dataclass
class BatchData:
    """Batch of training data with tensors."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[int]
    feature_indices: list[int]


# ==============================================================================
# 4. MODEL UTILITIES
# ==============================================================================


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


def build_training_prompt(positive_negative_examples: bool) -> str:
    """Build the training prompt for SAE explanations."""
    if positive_negative_examples:
        question = """Can you explain to me the concept of what 'X' means? Give positive and negative examples of what the concept would activate on. Format your final answer with <explanation>."""
    else:
        question = X_PROMPT
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
) -> list[TrainingDataPoint]:
    """Every prompt is exactly the same - the only difference is the steering vectors."""

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
        single_steering_vector = data_point.steering_vectors[0]

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
    sae: BaseSAE,
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


def load_model(
    cfg: SelfInterpTrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
    use_lora: bool,
) -> AutoModelForCausalLM:
    print(f"Loading model: {cfg.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map="auto", torch_dtype=dtype, attn_implementation="eager"
    )

    if use_lora:
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

    return model


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
    sae: BaseSAE,
    device: torch.device,
    dtype: torch.dtype,
    global_step: int,
):
    """Run evaluation and save results."""
    model.eval()
    with torch.no_grad():
        all_feature_results_this_eval_step = []
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
                sae=sae,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
            )
            all_feature_results_this_eval_step.extend(feature_results)

        save_logs(
            eval_results_path="eval_logs.json",
            global_step=global_step,
            all_feature_results_this_eval_step=all_feature_results_this_eval_step,
        )


def train_model(
    cfg: SelfInterpTrainingConfig,
    training_data: list[TrainingDataPoint],
    eval_data: list[TrainingDataPoint],
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    submodule: torch.nn.Module,
    sae: BaseSAE,
    device: torch.device,
    dtype: torch.dtype,
    verbose: bool = False,
):
    max_grad_norm = 1.0
    run_name = f"{cfg.model_name}-layer{cfg.sae_layer}-decoder-shorter-prompt"

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    total_training_steps = cfg.num_epochs * len(training_data)
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
            clip_grad_norm_(model.parameters(), max_grad_norm)
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
                    sae=sae,
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
                        commit_message=f"SAE introspection LoRA - {run_name} - step {global_step}",
                    )
                    print("Pushed LoRA adapter to Hugging Face Hub.")

            global_step += 1

    print("Training complete.")

    # Final evaluation
    print("Running final evaluation...")
    run_evaluation(
        cfg=cfg,
        eval_data=eval_data,
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        sae=sae,
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
            commit_message=f"SAE introspection LoRA - {run_name} - final model",
            private=cfg.hf_private_repo,
        )

    # wandb finishing is handled in main()


def main(
    explanations_file: str,
    model_name: str,
    sae_repo_id: str,
    hook_layer: int,
    hf_repo_name: Optional[str] = None,
):
    """Main script logic."""

    # Set up Hugging Face login at the start
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

    explanations: list[SAEExplained] = load_explanations_from_jsonl(explanations_file)
    cfg = SelfInterpTrainingConfig(
        # Model settings
        model_name=model_name,
        train_batch_size=4,
        eval_batch_size=128,  # 8 * 16
        # SAE settings
        sae_repo_id=sae_repo_id,
        sae_layer=9,
        hook_onto_layer=hook_layer,
        sae_width=131,
        # Experiment settings
        eval_set_size=100,
        use_decoder_vectors=True,
        generation_kwargs={
            "do_sample": True,
            "temperature": 1.0,
            "max_new_tokens": 600,
        },
        steering_coefficient=2.0,
        # LoRA settings
        use_lora=True,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules="all-linear",
        # Training settings
        lr=2e-5,
        eval_steps=99999999,
        num_epochs=1,
        save_steps=int(2000 / 4),  # save every 2000 samples
        # num_epochs=4,
        # save every epoch
        # save_steps=math.ceil(len(explanations) / 4),
        save_dir="checkpoints",
        # Hugging Face settings - set these based on your needs
        hf_push_to_hub=True,  # Only enable if login successful
        hf_repo_id=hf_repo_id_computed,
        hf_private_repo=False,  # Set to False if you want public repo
        positive_negative_examples=False,
    )

    print(asdict(cfg))
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Initialize wandb and upload the explanations file as an artifact at script start
    wandb_project = "sae_introspection"
    run_name = f"{cfg.model_name}-layer{cfg.sae_layer}-decoder-shorter-prompt"
    wandb.init(project=wandb_project, name=run_name, config=asdict(cfg))

    artifact_base = os.path.splitext(os.path.basename(explanations_file))[0]
    explanations_artifact = wandb.Artifact(
        name=f"explanations-{artifact_base}",
        type="dataset",
        description="SAE explanations JSONL used for training",
    )
    explanations_artifact.add_file(explanations_file)
    wandb.run.log_artifact(explanations_artifact)

    training_examples = [
        TrainingExample.with_positive_and_negative_examples(exp)
        if cfg.positive_negative_examples
        else TrainingExample.with_explanation_only(exp)
        for exp in explanations
    ]

    print(f"Loaded {len(training_examples)} training examples from {explanations_file}")

    model = load_model(cfg, device, dtype, use_lora=cfg.use_lora)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    sae = load_sae(
        sae_repo_id=cfg.sae_repo_id,
        sae_filename=cfg.sae_filename,
        sae_layer=cfg.sae_layer,
        model_name=cfg.model_name,
        device=device,
        dtype=dtype,
    )
    submodule = get_submodule(model, cfg.hook_onto_layer, cfg.use_lora)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_features = set()

    for example in training_examples:
        train_features.add(example.feature_idx)

    # For evaluation, we'll use a subset of the training features
    # In a real scenario, you might want to load a separate eval set
    print(f"train examples: {len(training_examples)}")
    print(f"Train features: {len(train_features)}")

    # Use provided eval features unless empty, then set a default
    if not cfg.eval_features:
        cfg.eval_features = [i for i in range(10)] + [i for i in range(20_000, 20_020)]

    # Respect eval_set_size by slicing the features list
    selected_eval_features = cfg.eval_features
    if cfg.eval_set_size and cfg.eval_set_size > 0:
        selected_eval_features = cfg.eval_features[: cfg.eval_set_size]

    print(f"Using {len(selected_eval_features)} features for evaluation")

    train_eval_prompt = build_training_prompt(cfg.positive_negative_examples)

    training_data: list[TrainingDataPoint] = construct_train_dataset(
        cfg,
        len(training_examples),
        # dataset_size,
        train_eval_prompt,
        training_examples,
        sae,
        tokenizer,
    )

    eval_data = construct_eval_dataset(
        cfg,
        len(selected_eval_features),
        train_eval_prompt,
        selected_eval_features,
        {},  # Empty dict since we don't use api_data anymore
        sae,
        tokenizer,
    )

    print(f"training data: {len(training_data)}, eval data: {len(eval_data)}")

    train_model(
        cfg,
        training_data,
        eval_data,
        model,
        tokenizer,
        submodule,
        sae,
        device,
        dtype,
        verbose=True,
    )

    wandb.finish()


if __name__ == "__main__":
    # main(
    #     explanations_file="data/20aug_sae_sfted_gpt-5-mini-2025-08-07.jsonl",
    #     hf_repo_name="gemma-hook-layer-0",
    #     model_name="google/gemma-2-9b-it",
    #     sae_repo_id="google/gemma-scope-9b-it-res",
    # )
    main(
        explanations_file="data/10k_qwen_28aug_sae_sfted_gpt-5-mini-2025-08-07.jsonl",
        hf_repo_name="qwen-hook-layer-1-2ndsep",
        model_name="Qwen/Qwen3-8B",
        hook_layer=1,
        sae_repo_id="adamkarvonen/qwen3-8b-saes",
    )
