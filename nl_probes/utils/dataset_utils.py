import json

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from transformers import AutoTokenizer

from detection_eval.detection_basemodels import SAEInfo


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
