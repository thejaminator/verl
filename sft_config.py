import contextlib
import datetime
import json
from dataclasses import dataclass, field
from typing import Any

import torch
from huggingface_hub import login, whoami
from pydantic import BaseModel, ConfigDict, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

from detection_eval.detection_basemodels import SAEInfo


@dataclass
class SelfInterpTrainingConfig:
    # --- Model ---
    model_name: str = "Qwen/Qwen3-8B"
    hook_onto_layer: int = 1
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])
    act_layers: list[int] = field(default_factory=list)  # derived if empty

    # --- Data / experiment ---
    sae_sft_datasets: list[str] = field(default_factory=list)  # pass in or compute outside
    classification_train_datasets: list[str] = field(default_factory=list)
    classification_eval_datasets: list[str] = field(default_factory=list)
    additional_train_dataset_filenames: list[str] = field(default_factory=list)
    use_decoder_vectors: bool = True
    generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"do_sample": True, "temperature": 1.0, "max_new_tokens": 300}
    )
    steering_coefficient: float = 2.0
    min_act_collect_offset: int = -2
    max_act_collect_offset: int = -5
    max_sae_sft_examples: int = 50_000
    max_classification_examples: int = 10_000
    test_set_size_per_ds: int = 25
    dataset_folder: str = "sft_training_data"
    num_qa_per_sample: int = 3

    # --- Batching ---
    train_batch_size: int = 16
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
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    eval_steps: int = 9_999_999  # effectively off by default
    eval_on_start: bool = False
    gradient_checkpointing: bool = False
    window_mult: int = 20
    save_steps: int = 5_000
    save_dir: str = "checkpoints"
    seed: int = 42
    eval_logs_path: str = "eval_logs.json"
    load_lora_path: str | None = None

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
        raise NotImplementedError("Not implemented")
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
        prompt = f"{sae_explanation.explanation}"
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


def layer_percent_to_layer(model_name: str, layer_percent: int) -> int:
    """Convert a layer percent to a layer number."""
    if model_name == "Qwen/Qwen3-8B":
        max_layers = 36
        return int(max_layers * (layer_percent / 100))
    else:
        raise ValueError(f"Unknown model name: {model_name}")


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


def construct_batch(
    training_data: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
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


def find_pattern_in_tokens(token_ids: list[int], pattern: str, tokenizer: AutoTokenizer) -> int | None:
    """
    Find the starting position of a pattern in a list of token IDs.

    Returns the index of the first token of the pattern, or None if not found.
    """
    start_idx = 0
    end_idx = len(token_ids)

    x_token_id = tokenizer.encode("X", add_special_tokens=False)[0]

    for i in range(start_idx, end_idx):
        if token_ids[i] == x_token_id:
            surrounding_tokens = token_ids[i - 2 : i + 2]
            surrounding_text = tokenizer.decode(surrounding_tokens)
            assert pattern in surrounding_text, (
                f"Expected pattern {pattern} in {surrounding_text}, got {surrounding_text}"
            )
            return i

    return None


def create_training_datapoint(
    prompt: str, target_response: str, tokenizer: AutoTokenizer, acts_D: torch.Tensor, feature_idx: int
) -> TrainingDataPoint:
    input_messages = [{"role": "user", "content": prompt}]

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

    position = find_pattern_in_tokens(full_prompt_ids, "<<X>>", tokenizer)
    if position is None:
        print(f"Warning! Expected exactly one X token, got {position}, classification prompt: {prompt}")
        raise ValueError("Expected exactly one X token")
    # May want to support multiple X tokens in the future
    positions = [position]
    steering_vectors = [acts_D.cpu().clone().detach()]

    training_data_point = TrainingDataPoint(
        input_ids=full_prompt_ids,
        labels=labels,
        steering_vectors=steering_vectors,
        positions=positions,
        feature_idx=feature_idx,
        target_output=target_response,
    )

    return training_data_point


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
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.

    Args:
        model: The model to run.
        submodule: The submodule to hook into.
        inputs_BL: The inputs to the model.
        use_no_grad: Whether to run the forward pass within a `torch.no_grad()` context. Defaults to True.
    """
    activations_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        # For many models, the submodule outputs are a tuple or a single tensor:
        # If "outputs" is a tuple, pick the relevant item:
        #   e.g. if your layer returns (hidden, something_else), you'd do outputs[0]
        # Otherwise just do outputs
        if isinstance(outputs, tuple):
            activations_BLD = outputs[0]
        else:
            activations_BLD = outputs

        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(gather_target_act_hook)

    # Determine the context manager based on the flag
    context_manager = torch.no_grad() if use_no_grad else contextlib.nullcontext()

    try:
        # Use the selected context manager
        with context_manager:
            _ = model(**inputs_BL)  # type: ignore
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    return activations_BLD  # type: ignore


def collect_activations_multiple_layers(
    model: AutoModelForCausalLM,
    submodules: dict[int, torch.nn.Module],
    inputs_BL: dict[str, torch.Tensor],
    min_offset: int | None,
    max_offset: int | None,
) -> dict[int, torch.Tensor]:
    if min_offset is not None:
        assert max_offset is not None, "max_offset must be provided if min_offset is provided"
        assert max_offset < min_offset, "max_offset must be less than min_offset"
        assert min_offset < 0, "min_offset must be less than 0"
        assert max_offset < 0, "max_offset must be less than 0"
    else:
        assert max_offset is None, "max_offset must be provided if min_offset is not provided"

    activations_BLD_by_layer = {}

    module_to_layer = {submodule: layer for layer, submodule in submodules.items()}

    max_layer = max(submodules.keys())

    def gather_target_act_hook(module, inputs, outputs):
        layer = module_to_layer[module]

        if isinstance(outputs, tuple):
            activations_BLD_by_layer[layer] = outputs[0]
        else:
            activations_BLD_by_layer[layer] = outputs

        if min_offset is not None:
            activations_BLD_by_layer[layer] = activations_BLD_by_layer[layer][:, max_offset:min_offset, :]

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
