import contextlib
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import torch
from openai import BaseModel

from detection_eval.detection_basemodels import SAE, SAEActivations

X_PROMPT = "Can you explain to me what 'X' means? Format your final answer with <explanation>"


def get_vllm_steering_hook(
    vectors: list[torch.Tensor],
    positions: list[int],
    prompt_lengths: list[int],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Debug version of your steering hook with detailed logging
    """
    vec_BD = torch.stack(vectors)  # (B, d_model)
    pos_B = torch.tensor(positions, dtype=torch.long)  # (B,)
    B, d_model = vec_BD.shape
    vec_BD = vec_BD.to(device, dtype)
    pos_B = pos_B.to(device)

    def hook_fn(module, _input, output):
        # passed prompt lengths should line up hopefully
        tokens_L = _input[0]

        if tokens_L.shape[0] == B:
            # means we are in decoding, not prefill. So no need to steer.
            return output

        # if there aren't any 0s in tokens_L, then we are NOT in prefill. So skip
        if not torch.any(tokens_L == 0):
            return output

        count = 0
        for prompt_length in prompt_lengths:
            expected_position_indices_L = torch.arange(prompt_length, device=device)
            try:
                assert tokens_L[count : count + prompt_length].equal(expected_position_indices_L), (
                    f"Position indices mismatch at index {count}, expected {expected_position_indices_L}, got {tokens_L[count : count + prompt_length]}"
                )
            except AssertionError as e:
                breakpoint()
                raise e

            count += prompt_length

        before_resid_flat, resid_flat, *rest = output

        assert count == tokens_L.shape[0]
        assert resid_flat.shape[0] == tokens_L.shape[0]
        assert resid_flat.shape[1] == d_model

        intervention_indices_L = []
        idx = 0

        for i in range(len(prompt_lengths)):
            intervention_idx = torch.tensor(idx + positions[i], device=device)
            intervention_indices_L.append(intervention_idx)
            idx += prompt_lengths[i]

        assert idx >= tokens_L.shape[0]

        intervention_indices_L = torch.stack(intervention_indices_L)

        assert intervention_indices_L.shape[0] == B

        orig_BD = resid_flat[intervention_indices_L]

        assert orig_BD.shape == (B, d_model)

        # Compute norms and steering
        norms_B1 = orig_BD.norm(dim=-1, keepdim=True).detach()
        normalized_features = torch.nn.functional.normalize(vec_BD, dim=-1)
        steered_BD = normalized_features * norms_B1 * steering_coefficient

        print(f"  Normalized feature norms: {normalized_features.norm(dim=-1).tolist()}")
        print(f"  Original norms: {norms_B1.squeeze().tolist()}")
        print(f"  Steered activation norms: {steered_BD.norm(dim=-1).tolist()}")

        # Calculate the change magnitude BEFORE applying
        change_magnitude = (steered_BD - orig_BD).norm(dim=-1)
        print(f"  Change magnitudes: {change_magnitude.tolist()}")

        if change_magnitude.max() < 1e-4:
            print("  ⚠️  WARNING: Very small change magnitude!")

        # Apply the steering
        print(f"  Applying steering at positions: {pos_B.tolist()}")
        resid_flat[intervention_indices_L] = steered_BD

        return (before_resid_flat, resid_flat, *rest)

    return hook_fn


@contextlib.contextmanager
def add_hook(
    module: torch.nn.Module,
    hook: Callable,
):
    """Temporarily adds a forward hook to a model module.

    Args:
        module: The PyTorch module to hook
        hook: The hook function to apply

    Yields:
        None: Used as a context manager

    Example:
        with add_hook(model.layer, hook_fn):
            output = model(input)
    """
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def get_hf_activation_steering_hook(
    vectors: list[torch.Tensor],  # [B, d_model]
    positions: list[int],  # [B]
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    HF hook with debug prints to compare against vLLM
    """
    # ---- pack Python lists → torch tensors once, outside the hook ----
    vec_BD = torch.stack(vectors).to(device, dtype)  # (B, d)
    pos_B = torch.tensor(positions, dtype=torch.long, device=device)  # (B,)
    B, d_model = vec_BD.shape

    # Handle the case where positions might create a scalar tensor instead of a 1D tensor
    if pos_B.dim() == 0:  # scalar tensor case (when positions is a single integer instead of list)
        pos_B = pos_B.unsqueeze(0)  # Convert scalar to (1,) shape

    assert pos_B.shape == (B,)
    vec_BD = vec_BD.to(device, dtype)
    pos_B = pos_B.to(device)

    # print("🔧 HF STEERING HOOK SETUP:")
    # print(f"  Batch size: {B}")
    # print(f"  Feature vector shape: {vec_BD.shape}")
    # print(f"  Positions: {pos_B.tolist()}")
    # print(f"  Steering coefficient: {steering_coefficient}")
    # print(f"  Feature vector norms: {vec_BD.norm(dim=-1).tolist()}")

    def hook_fn(module, _input, output):
        resid_BLD, *rest = output  # Gemma returns (resid, hidden_states, ...)
        B_actual, L, d_model_actual = resid_BLD.shape

        # Only touch the *prompt* forward pass (sequence length > 1)
        if L <= 1:
            return (resid_BLD, *rest)

        # Safety: make sure every position is inside current sequence
        if (pos_B >= L).any():
            bad = pos_B[pos_B >= L].min().item()
            raise IndexError(f"position {bad} is out of bounds for length {L}")

        # print("\n🎯 HF STEERING HOOK EXECUTING:")
        # print(f"  Module: {type(module).__name__}")
        # print(f"  Input shape: {resid_BLD.shape}")
        # print(f"  Sequence length: {L}")
        # print(f"  Expected batch size: {B}, actual: {B_actual}")

        # ---- compute norms of original activations at the target slots ----
        batch_idx_B = torch.arange(B, device=device)  # (B,)
        orig_BD = resid_BLD[batch_idx_B, pos_B]  # (B, d)

        norms_B1 = orig_BD.norm(dim=-1, keepdim=True).detach()  # (B, 1)

        # ---- build steered vectors ----
        normalized_features = torch.nn.functional.normalize(vec_BD, dim=-1)
        steered_BD = normalized_features * norms_B1 * steering_coefficient  # (B, d)

        # Calculate the change magnitude BEFORE applying
        change_magnitude = (steered_BD - orig_BD).norm(dim=-1)

        # sometiems this blows up. not sure why.
        # if change_magnitude.max() < 1e-4:
        #     print("WARNING: Very small change magnitude in get_hf_activation_steering_hook")
        #     raise ValueError("Very small change magnitude!")

        # ---- in-place replacement via advanced indexing ----
        resid_BLD[batch_idx_B, pos_B] = steered_BD

        return (resid_BLD, *rest)

    return hook_fn


class SAEVerlData(BaseModel):
    sae_id: int
    feature_vector: list[float]  # This needs to be added in by the script
    position_id: int  # This needs to be added in by the script
    activations: SAEActivations  # For reward model.
    hard_negatives: list[SAEActivations]  # For reward model.


class SAEVerlDataTypedDict(TypedDict):
    """Typed dict that gets passed around in verl"""

    sae_id: int
    feature_vector: list[float]  # This needs to be added in by the script
    position_id: int  # This needs to be added in by the script
    activations: dict[str, Any]
    hard_negatives: list[dict[str, Any]]


def make_sae_verl_typed_dict(sae_data: SAE, position_id: int, feature_vector: list[float]) -> SAEVerlDataTypedDict:
    return {
        "sae_id": sae_data.sae_id,
        "position_id": position_id,
        "feature_vector": feature_vector,
        "activations": sae_data.activations.model_dump(),
        "hard_negatives": [m.model_dump() for m in sae_data.hard_negatives],
    }


@dataclass(kw_only=True)
class HookArgs:
    vectors: list[torch.Tensor]
    positions: list[int]
    steering_coefficient: float


def verl_data_to_hook_args(verl_data: list[SAEVerlDataTypedDict], device: torch.device) -> HookArgs:
    feature_vectors = [m["feature_vector"] for m in verl_data]
    positions = [m["position_id"] for m in verl_data]
    vectors = [torch.tensor(fv, dtype=torch.bfloat16, device=device) for fv in feature_vectors]
    steering_coefficient = 2
    return HookArgs(vectors=vectors, positions=positions, steering_coefficient=steering_coefficient)
