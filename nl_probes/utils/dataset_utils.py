import json

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

from detection_eval.detection_basemodels import SAEInfo
from detection_eval.steering_hooks import SPECIAL_TOKEN, get_introspection_prefix
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule


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
    """Training data point with tensors.
    If steering_vectors is None, then we calculate the steering vectors on the fly
    from the context_input_ids and context_positions."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    datapoint_type: str
    input_ids: list[int]
    labels: list[int]  # Can contain -100 for ignored tokens
    layer: int
    steering_vectors: torch.Tensor | None
    positions: list[int]
    feature_idx: int
    target_output: str
    context_input_ids: list[int] | None
    context_positions: list[int] | None

    @model_validator(mode="after")
    def _check_context_alignment(cls, values):
        sv = values.steering_vectors
        if sv is not None:
            if len(values.positions) != sv.shape[0]:
                raise ValueError("positions and steering_vectors must have the same length")
            if values.context_positions is not None or values.context_input_ids is not None:
                raise ValueError("context_* must be None when steering_vectors is provided")
        else:
            if values.context_positions is None or values.context_input_ids is None:
                raise ValueError("context_* must be provided when steering_vectors is None")
            if len(values.positions) != len(values.context_positions):
                raise ValueError("positions and context_positions must have the same length")
        return values


class BatchData(BaseModel):
    """Batch of training data with tensors."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[list[int]]
    feature_indices: list[int]


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

        for i in range(len(data_point.positions)):
            data_point.positions[i] += padding_length

        if data_point.steering_vectors is not None:
            data_point.steering_vectors = data_point.steering_vectors.to(device)

        batch_positions.append(data_point.positions)
        batch_steering_vectors.append(data_point.steering_vectors)
        batch_feature_indices.append(data_point.feature_idx)

    return BatchData(
        input_ids=torch.stack(batch_tokens),
        labels=torch.stack(batch_labels),
        attention_mask=torch.stack(batch_attn_masks),
        steering_vectors=batch_steering_vectors,
        positions=batch_positions,
        feature_indices=batch_feature_indices,
    )


def materialize_missing_steering_vectors(
    batch_points: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> None:
    """
    In-place materialization of missing steering vectors for a heterogenous batch
    where different items can request activations from different layers.

    Steps:
      1) Find items with steering_vectors=None.
      2) Build a left-padded batch from their context_input_ids.
      3) Register hooks for all unique requested layers and run exactly one forward pass.
      4) For each item, take activations at its requested layer and its context_positions,
         then write back a [num_positions, D] CPU tensor to dp.steering_vectors.

    No-op if every item already has steering_vectors.
    """
    # Select datapoints that need generation
    to_fill: list[tuple[int, TrainingDataPoint]] = [
        (i, dp) for i, dp in enumerate(batch_points) if dp.steering_vectors is None
    ]
    if not to_fill:
        return

    # Validate context fields
    for _, dp in to_fill:
        if dp.context_input_ids is None or dp.context_positions is None:
            raise ValueError(
                "Datapoint has steering_vectors=None but is missing context_input_ids or context_positions"
            )

    # Build the input batch (left padding to match your construct_batch convention)
    pad_id = tokenizer.pad_token_id
    contexts: list[list[int]] = [list(dp.context_input_ids) for _, dp in to_fill]
    positions_per_item: list[list[int]] = [list(dp.context_positions) for _, dp in to_fill]
    max_len = max(len(c) for c in contexts)

    input_ids_tensors: list[torch.Tensor] = []
    attn_masks_tensors: list[torch.Tensor] = []
    left_offsets: list[int] = []

    device = next(model.parameters()).device

    for c in contexts:
        pad_len = max_len - len(c)
        input_ids_tensors.append(torch.tensor([pad_id] * pad_len + c, dtype=torch.long, device=device))
        # For HF, bool masks are fine; your construct_batch uses bool too
        attn_masks_tensors.append(torch.tensor([False] * pad_len + [True] * len(c), dtype=torch.bool, device=device))
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_tensors, dim=0),
        "attention_mask": torch.stack(attn_masks_tensors, dim=0),
    }

    # Prepare hooks for all unique requested layers
    layers_needed = sorted({dp.layer for _, dp in to_fill})
    submodules = {layer: get_hf_submodule(model, layer, use_lora=False) for layer in layers_needed}

    # Run a single pass with dropout off, then restore the previous train/eval mode
    was_training = model.training
    model.eval()
    model.disable_adapters()
    # [layer] -> [B, L, D], where B == len(to_fill)
    acts_by_layer = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_BL,
        min_offset=None,
        max_offset=None,
    )
    if was_training:
        model.train()
    model.enable_adapters()

    # Slice the right positions per item and write back
    B = len(to_fill)
    for b in range(B):
        _, dp = to_fill[b]
        layer = dp.layer
        if layer not in acts_by_layer:
            raise KeyError(f"No activations captured for requested layer {layer}")
        acts_BLD = acts_by_layer[layer]  # [B, L, D]
        idxs = [p + left_offsets[b] for p in positions_per_item[b]]

        # Bounds check for safety
        L = acts_BLD.shape[1]
        if any(i < 0 or i >= L for i in idxs):
            raise IndexError(f"Activation index out of range for item {b}: {idxs} with L={L}")

        vectors = acts_BLD[b, idxs, :].detach()  # [num_positions, D]
        assert len(vectors.shape) == 2, f"Expected 2D tensor, got vectors.shape={vectors.shape}"
        dp.steering_vectors = vectors.to(device)


def find_pattern_in_tokens(
    token_ids: list[int], special_token_str: str, num_positions: int, tokenizer: AutoTokenizer
) -> list[int]:
    start_idx = 0
    end_idx = len(token_ids)
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    assert len(special_token_id) == 1, f"Expected single token, got {len(special_token_id)}"
    special_token_id = special_token_id[0]
    positions = []

    for i in range(start_idx, end_idx):
        if len(positions) == num_positions:
            break
        if token_ids[i] == special_token_id:
            positions.append(i)

    assert len(positions) == num_positions, f"Expected {num_positions} positions, got {len(positions)}"
    assert positions[-1] - positions[0] == num_positions - 1, f"Positions are not consecutive: {positions}"

    final_pos = positions[-1] + 1
    final_tokens = token_ids[final_pos : final_pos + 2]
    final_str = tokenizer.decode(final_tokens, skip_special_tokens=False)
    assert "\n" in final_str, f"Expected newline in {final_str}"

    return positions


def create_training_datapoint(
    datapoint_type: str,
    prompt: str,
    target_response: str,
    layer: int,
    num_positions: int,
    tokenizer: AutoTokenizer,
    acts_BD: torch.Tensor | None,
    feature_idx: int,
    context_input_ids: list[int] | None = None,
    context_positions: list[int] | None = None,
) -> TrainingDataPoint:
    prefix = get_introspection_prefix(layer, num_positions)
    assert prefix not in prompt, f"Prefix {prefix} found in prompt {prompt}"
    prompt = prefix + prompt
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

    positions = find_pattern_in_tokens(full_prompt_ids, SPECIAL_TOKEN, num_positions, tokenizer)

    if acts_BD is None:
        assert context_input_ids is not None and context_positions is not None, (
            "acts_BD is None but context_input_ids and context_positions are None"
        )
    else:
        assert len(acts_BD.shape) == 2, f"Expected 2D tensor, got {acts_BD.shape}"
        acts_BD = acts_BD.cpu().clone().detach()
        assert len(positions) == acts_BD.shape[0], f"Expected {acts_BD.shape[0]} positions, got {len(positions)}"

    training_data_point = TrainingDataPoint(
        input_ids=full_prompt_ids,
        labels=labels,
        layer=layer,
        steering_vectors=acts_BD,
        positions=positions,
        feature_idx=feature_idx,
        target_output=target_response,
        datapoint_type=datapoint_type,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
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
