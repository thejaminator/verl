import datetime
from dataclasses import dataclass, field
from typing import Any

import torch
from huggingface_hub import login, whoami
from pydantic import BaseModel, ConfigDict, field_validator
from transformers import AutoTokenizer

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
    use_decoder_vectors: bool = True
    generation_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"do_sample": True, "temperature": 1.0, "max_new_tokens": 300}
    )
    steering_coefficient: float = 2.0
    act_collect_offset: int = -3
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
    lr: float = 1e-5
    max_grad_norm: float = 1.0
    eval_steps: int = 9_999_999  # effectively off by default
    eval_on_start: bool = False
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
