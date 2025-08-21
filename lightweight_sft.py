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

# All necessary imports are now included above
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import safetensors.torch
import torch
import torch.nn as nn
import wandb
from huggingface_hub import hf_hub_download, login, whoami
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import PreTrainedTokenizer

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

    print(f"Successfully pushed LoRA adapter to: https://huggingface.co/{repo_id}")


# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================


def get_sae_info(sae_repo_id: str) -> tuple[int, int, int, str]:
    sae_layer = 9
    sae_layer_percent = 25

    if sae_repo_id == "google/gemma-scope-9b-it-res":
        sae_width = 131

        if sae_width == 16:
            sae_filename = f"layer_{sae_layer}/width_16k/average_l0_88/params.npz"
        elif sae_width == 131:
            sae_filename = f"layer_{sae_layer}/width_131k/average_l0_121/params.npz"
        else:
            raise ValueError(f"Unknown SAE width: {sae_width}")
    elif sae_repo_id == "fnlp/Llama3_1-8B-Base-LXR-32x":
        sae_width = 32
        sae_filename = ""
    else:
        raise ValueError(f"Unknown SAE repo ID: {sae_repo_id}")
    return sae_width, sae_layer, sae_layer_percent, sae_filename


@dataclass
class SelfInterpTrainingConfig:
    """Configuration settings for the script."""

    # --- Model Settings ---
    model_name: str
    train_batch_size: int
    eval_batch_size: int

    # --- SAE (Sparse Autoencoder) Settings ---
    sae_repo_id: str
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
        self.sae_width, self.sae_layer, self.sae_layer_percent, self.sae_filename = get_sae_info(self.sae_repo_id)


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


def get_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule"""
    model_name = model.config._name_or_path

    if use_lora:
        if "pythia" in model_name:
            raise ValueError("Need to determine how to get submodule for LoRA")
        elif "gemma" in model_name or "mistral" in model_name or "Llama" in model_name:
            return model.base_model.model.model.layers[layer]
        else:
            raise ValueError(f"Please add submodule for model {model_name}")

    if "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    elif "gemma" in model_name or "mistral" in model_name or "Llama" in model_name:
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


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
# 5. SAE CLASSES
# ==============================================================================


class BaseSAE(nn.Module, ABC):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        super().__init__()

        # Required parameters
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))

        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Required attributes
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.hook_layer = hook_layer

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        self.to(dtype=self.dtype, device=self.device)

    @abstractmethod
    def encode(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @abstractmethod
    def decode(self, feature_acts: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError("Encode method must be implemented by child classes")

    def to(self, *args, **kwargs):
        """Handle device and dtype updates"""
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self

    @torch.no_grad()
    def check_decoder_norms(self) -> bool:
        """
        It's important to check that the decoder weights are normalized.
        """
        norms = torch.norm(self.W_dec, dim=1).to(dtype=self.dtype, device=self.device)

        # In bfloat16, it's common to see errors of (1/256) in the norms
        tolerance = 1e-2 if self.W_dec.dtype in [torch.bfloat16, torch.float16] else 1e-5

        if torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
            return True
        else:
            max_diff = torch.max(torch.abs(norms - torch.ones_like(norms)))
            print(f"Decoder weights are not normalized. Max diff: {max_diff.item()}")
            return False


class JumpReluSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        self.threshold = nn.Parameter(torch.zeros(d_sae, dtype=dtype, device=device))
        self.d_sae = d_sae
        self.d_in = d_in

    def encode(self, x: torch.Tensor):
        pre_acts = x @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, feature_acts: torch.Tensor):
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


class TopKSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        use_threshold: bool = False,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        assert isinstance(k, int) and k > 0
        self.register_buffer("k", torch.tensor(k, dtype=torch.int, device=device))
        self.d_sae = d_sae
        self.d_in = d_in
        self.pre_encoder_bias = False

        self.use_threshold = use_threshold
        if use_threshold:
            # Optional global threshold to use during inference. Must be positive.
            self.register_buffer("threshold", torch.tensor(-1.0, dtype=dtype, device=device))

    def encode(self, x: torch.Tensor):
        """Note: x can be either shape (B, F) or (B, L, F)"""
        pre_acts = x @ self.W_enc + self.b_enc

        # Get top-k activations
        top_acts, top_indices = torch.topk(pre_acts, self.k.item(), dim=-1)

        # Create sparse representation
        acts = torch.zeros_like(pre_acts)
        if len(acts.shape) == 2:  # (B, F)
            batch_indices = torch.arange(acts.shape[0], device=acts.device).unsqueeze(1)
            acts[batch_indices, top_indices] = torch.nn.functional.relu(top_acts)
        else:  # (B, L, F)
            batch_indices = torch.arange(acts.shape[0], device=acts.device).unsqueeze(1).unsqueeze(2)
            seq_indices = torch.arange(acts.shape[1], device=acts.device).unsqueeze(0).unsqueeze(2)
            acts[batch_indices, seq_indices, top_indices] = torch.nn.functional.relu(top_acts)

        if self.use_threshold and hasattr(self, "threshold"):
            acts = acts * (acts > self.threshold)

        return acts

    def decode(self, feature_acts: torch.Tensor):
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon


# ==============================================================================
# 6. UTILITY FUNCTIONS
# ==============================================================================


def build_training_prompt(positive_negative_examples: bool) -> str:
    """Build the training prompt for SAE explanations."""
    if positive_negative_examples:
        question = """Can you explain to me the concept of what 'X' means? Give positive and negative examples of what the concept would activate on. Format your final answer with <explanation>."""
    else:
        question = """Can you explain to me what 'X' means? Format your final answer with <explanation>"""
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


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_activation_steering_hook(
    vectors: list[torch.Tensor],  # [B, d_model]
    positions: list[int],  # [B]
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Returns a forward hook that *replaces* specified residual-stream activations
    during the initial prompt pass of `model.generate`.

    • vectors[b]  – feature vector to inject for batch b
    • positions[b]– token index (0-based, within prompt only) for batch b
    """

    # ---- pack Python lists → torch tensors once, outside the hook ----
    vec_BD = torch.stack(vectors)  # (B, d)
    pos_B = torch.tensor(positions, dtype=torch.long)  # (B,)

    B, d_model = vec_BD.shape
    assert pos_B.shape == (B,)

    vec_BD = vec_BD.to(device, dtype)
    pos_B = pos_B.to(device)

    def hook_fn(module, _input, output):
        resid_BLD, *rest = output  # Gemma returns (resid, hidden_states, ...)
        L = resid_BLD.shape[1]

        # Only touch the *prompt* forward pass (sequence length > 1)
        if L <= 1:
            return (resid_BLD, *rest)

        # Safety: make sure every position is inside current sequence
        if (pos_B >= L).any():
            bad = pos_B[pos_B >= L].min().item()
            raise IndexError(f"position {bad} is out of bounds for length {L}")

        # ---- compute norms of original activations at the target slots ----
        batch_idx_B = torch.arange(B, device=device)  # (B,)
        orig_BD = resid_BLD[batch_idx_B, pos_B]  # (B, d)
        norms_B1 = orig_BD.norm(dim=-1, keepdim=True).detach()  # (B, 1)

        # ---- build steered vectors ----
        steered_BD = torch.nn.functional.normalize(vec_BD, dim=-1) * norms_B1 * steering_coefficient  # (B, d)

        # ---- in-place replacement via advanced indexing ----
        resid_BLD[batch_idx_B, pos_B] = steered_BD

        return (resid_BLD, *rest)

    return hook_fn


# Note: collect_training_examples removed - we now read explanations from JSONL files


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
    hook_fn = get_activation_steering_hook(
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
    hook_fn = get_activation_steering_hook(
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
# 7. SAE LOADING FUNCTIONS
# ==============================================================================


def load_gemma_scope_jumprelu_sae(
    repo_id: str,
    filename: str,
    layer: int,
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    local_dir: str = "downloaded_saes",
) -> JumpReluSAE:
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )
    pytorch_path = path_to_params.replace(".npz", ".pt")

    # Doing this because npz files are often insanely slow to load
    if not os.path.exists(pytorch_path):
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
        torch.save(pt_params, pytorch_path)

    pt_params = torch.load(pytorch_path)

    d_in = pt_params["W_enc"].shape[0]
    d_sae = pt_params["W_enc"].shape[1]

    assert d_sae >= d_in

    sae = JumpReluSAE(d_in, d_sae, model_name, layer, device, dtype)
    sae.load_state_dict(pt_params)
    sae.to(dtype=dtype, device=device)

    normalized = sae.check_decoder_norms()
    if not normalized:
        raise ValueError("Decoder norms are not normalized. Implement a normalization method.")

    return sae


def load_llama_scope_topk_sae(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer: int,
    expansion_factor: int,
) -> TopKSAE:
    repo_id = f"fnlp/Llama3_1-8B-Base-LXR-{expansion_factor}x"
    config_filename = f"Llama3_1-8B-Base-L{layer}R-{expansion_factor}x/hyperparams.json"
    filename = f"Llama3_1-8B-Base-L{layer}R-{expansion_factor}x/checkpoints/final.safetensors"

    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir="downloaded_saes",
    )

    path_to_config = hf_hub_download(
        repo_id=repo_id,
        filename=config_filename,
        force_download=False,
        local_dir="downloaded_saes",
    )

    with open(path_to_config) as f:
        config = json.load(f)

    threshold = config["jump_relu_threshold"]
    k = config["top_k"]

    pt_params = safetensors.torch.load_file(path_to_params)

    key_mapping = {
        "encoder.weight": "W_enc",
        "decoder.weight": "W_dec",
        "encoder.bias": "b_enc",
        "decoder.bias": "b_dec",
    }

    renamed_params = {key_mapping.get(k, k): v for k, v in pt_params.items()}
    renamed_params["W_enc"] = renamed_params["W_enc"].T
    renamed_params["W_dec"] = renamed_params["W_dec"].T
    renamed_params["k"] = torch.tensor(k, dtype=torch.int, device=device)
    renamed_params["threshold"] = torch.tensor(threshold, dtype=dtype, device=device)

    sae = TopKSAE(
        d_in=renamed_params["b_dec"].shape[0],
        d_sae=renamed_params["b_enc"].shape[0],
        k=k,
        model_name=model_name,
        hook_layer=layer,
        device=device,
        dtype=dtype,
        use_threshold=True,
    )

    sae.load_state_dict(renamed_params)
    sae.to(device=device, dtype=dtype)

    # https://github.com/OpenMOSS/Language-Model-SAEs/blob/25180e32e82176924b62ab30a75fffd234260a9e/src/lm_saes/sae.py#L172
    # openmoss scaling strategy
    dataset_average_activation_norm = config["dataset_average_activation_norm"]
    input_norm_factor = sae.d_in**0.5 / dataset_average_activation_norm["in"]
    sae.b_enc.data /= input_norm_factor

    return sae


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


def load_sae(
    cfg: SelfInterpTrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> BaseSAE:
    # Note: There's some duplication here with saes.sae_loading_utils.py
    print(f"Loading SAE for layer {cfg.sae_layer} from {cfg.sae_repo_id}...")

    if cfg.sae_repo_id == "google/gemma-scope-9b-it-res":
        sae = load_gemma_scope_jumprelu_sae(
            repo_id=cfg.sae_repo_id,
            filename=cfg.sae_filename,
            layer=cfg.sae_layer,
            model_name=cfg.model_name,
            device=device,
            dtype=dtype,
        )
    elif cfg.sae_repo_id == "fnlp/Llama3_1-8B-Base-LXR-32x":
        sae = load_llama_scope_topk_sae(
            model_name=cfg.model_name,
            device=device,
            dtype=dtype,
            layer=cfg.sae_layer,
            expansion_factor=cfg.sae_width,
        )
    else:
        raise ValueError(f"Unknown SAE repo ID: {cfg.sae_repo_id}")

    return sae


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
    wandb_project = "sae_introspection"
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

            if global_step % cfg.save_steps == 0:
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


def main(explanations_file: str, hf_repo_name: Optional[str] = None):
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

    cfg = SelfInterpTrainingConfig(
        # Model settings
        model_name="google/gemma-2-9b-it",
        train_batch_size=4,
        eval_batch_size=128,  # 8 * 16
        # SAE settings
        sae_repo_id="google/gemma-scope-9b-it-res",
        sae_layer=9,
        sae_width=16,
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
        num_epochs=1,
        lr=2e-5,
        eval_steps=1000,
        save_steps=int(1000 / 4),  # save every 1000 samples
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

    explanations: list[SAEExplained] = load_explanations_from_jsonl(explanations_file)
    training_examples = [
        TrainingExample.with_positive_and_negative_examples(exp)
        if cfg.positive_negative_examples
        else TrainingExample.with_explanation_only(exp)
        for exp in explanations
    ]

    print(f"Loaded {len(training_examples)} training examples from {explanations_file}")

    model = load_model(cfg, device, dtype, use_lora=cfg.use_lora)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    sae = load_sae(cfg, device, dtype)
    submodule = get_submodule(model, cfg.sae_layer, cfg.use_lora)

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

    # Use a subset of training features for evaluation
    # 0 to 10, then 20_000 to 20_020
    # 0 to 10 is in training, 20_000 to 20_020 is in eval
    cfg.eval_features = [i for i in range(10)] + [i for i in range(20_000, 20_020)]

    print(f"Using {len(cfg.eval_features)} features for evaluation")

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
        len(cfg.eval_features),
        train_eval_prompt,
        cfg.eval_features,
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
    explanations_file = "20aug_sae_sfted_gpt-5-mini-2025-08-07.jsonl"
    main(explanations_file)
