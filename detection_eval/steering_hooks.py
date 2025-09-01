import contextlib
from dataclasses import dataclass
from typing import Callable

import torch

from detection_eval.detection_basemodels import SAEVerlDataTypedDict

def get_introspection_prompt(sae_layer: int) -> str:
    return f"Can you explain to me what 'X' means? It is from layer {sae_layer}. Format your final answer with <explanation>"


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

        number_of_zeroes = torch.sum(tokens_L == 0).item()
        # should be equal to number of prompts
        if number_of_zeroes != len(prompt_lengths):
            breakpoint()
            raise ValueError(
                f"Number of zeroes {number_of_zeroes} is not equal to number of prompt lengths {len(prompt_lengths)}"
            )

        count = 0
        for prompt_length in prompt_lengths:
            expected_position_indices_L = torch.arange(prompt_length, device=device)
            try:
                assert tokens_L[count : count + prompt_length].equal(expected_position_indices_L), (
                    f"Position indices mismatch at index {count}, expected {expected_position_indices_L}, got {tokens_L[count : count + prompt_length]}"
                )
            except AssertionError as e:
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

        # print(f"  Normalized feature norms: {normalized_features.norm(dim=-1).tolist()}")
        # print(f"  Original norms: {norms_B1.squeeze().tolist()}")
        # print(f"  Steered activation norms: {steered_BD.norm(dim=-1).tolist()}")

        # Calculate the change magnitude BEFORE applying
        change_magnitude = (steered_BD - orig_BD).norm(dim=-1)
        print(f"  Change magnitudes: {change_magnitude.tolist()}")

        if change_magnitude.max() < 1e-4:
            print("  ⚠️  WARNING: Very small change magnitude!")

        # Apply the steering
        # print(f"  Applying steering at positions: {pos_B.tolist()}")
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

    assert pos_B.shape == (B,)
    vec_BD = vec_BD.to(device, dtype)
    pos_B = pos_B.to(device)

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            # gemma
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            # qwen
            resid_BLD = output
            output_is_tuple = False
        # resid_BLD, *rest = output  # Gemma returns (resid, hidden_states, ...)
        B_actual, L, d_model_actual = resid_BLD.shape

        # Only touch the *prompt* forward pass (sequence length > 1)
        if L <= 1:
            if output_is_tuple:
                return (resid_BLD, *rest)
            else:
                return resid_BLD

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
        normalized_features = torch.nn.functional.normalize(vec_BD, dim=-1).detach()
        steered_BD = normalized_features * norms_B1 * steering_coefficient  # (B, d)

        # somehow verl explodes here and complains about dtype?
        steered_BD = steered_BD.to(dtype)

        # Calculate the change magnitude BEFORE applying
        # change_magnitude = (steered_BD - orig_BD).norm(dim=-1)

        # sometiems this blows up. not sure why.
        # if change_magnitude.max() < 1e-4:
        #     print("WARNING: Very small change magnitude in get_hf_activation_steering_hook")
        #     raise ValueError("Very small change magnitude!")

        # ---- in-place replacement via advanced indexing ----
        resid_BLD[batch_idx_B, pos_B] = steered_BD
        if output_is_tuple:
            return (resid_BLD, *rest)
        else:
            return resid_BLD

    return hook_fn


def get_rm_pad_log_probs_hook(
    vectors: list[torch.Tensor],  # [B, d_model]
    positions: list[int],  # [B]
    verl_positions: torch.Tensor,  # (1, total tokens).
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
    in_place: bool = False,  # Avoid doing inplace to allow compat with grad checkpointing
) -> Callable:
    """
    When verl has rmpad set to True (default?), the position_ids are (1, total tokens)
    example  of positions:
    tensor([[0, 1, 2, ,3 .....133, 134, 0, 1, 2, 3, ....101]])

    Use for calculating log probs in verl
    Note: Do not use for generating / rollouts. Only just for the forward pass for log probs.
    """
    # ---- pack Python lists → torch tensors once, outside the hook ----
    vec_BD = torch.stack(vectors).to(device, dtype)  # (B, d_model)
    B, d_model = vec_BD.shape
    pos_B = torch.tensor(positions, dtype=torch.long, device=device)  # (B,)

    def hook_fn(module, _input, output):
        try:
            # residual: (1, total_tokens, d_model)
            if isinstance(output, tuple):
                residual, *rest = output
                output_is_tuple = True
            else:
                residual = output
                output_is_tuple = False

            _, total_tokens, d_model_actual = residual.shape  # (1, L, d_model)
            expected_length = verl_positions.shape[1]
            assert total_tokens == expected_length, f"Expected length {expected_length}, got {total_tokens}"

            # Identify the start index of each sequence in the flattened positions.
            # verl_positions: (1, total_tokens)
            seq_start_indices = (verl_positions[0] == 0).nonzero(as_tuple=True)[0]  # (B,)
            assert seq_start_indices.numel() == B, (
                f"Expected {B} sequence starts (zeros) in verl_positions, got {seq_start_indices.numel()}"
            )

            # Validate that each requested local position fits inside its sequence span.
            # Sequence ends are next start, or total_tokens for the last sequence
            seq_end_indices = torch.cat(
                [seq_start_indices[1:], torch.tensor([total_tokens], device=device, dtype=torch.long)],
                dim=0,
            )  # (B,)
            max_local_lengths = seq_end_indices - seq_start_indices  # (B,)
            assert torch.all(pos_B < max_local_lengths), (
                f"Some positions exceed their sequence lengths: positions={pos_B.tolist()}, lengths={max_local_lengths.tolist()}"
            )

            # Map each local position to its global token index within the flattened residual
            # global_indices: (B,)
            global_indices = seq_start_indices + pos_B

            # ---- compute norms of original activations at the target slots ----
            # orig_BD: (B, d_model)
            orig_BD = residual[0, global_indices, :]
            norms_B1 = orig_BD.norm(dim=-1, keepdim=True).detach()  # (B, 1)

            # ---- build steered vectors ----
            # normalized_features: (B, d_model)
            normalized_features = torch.nn.functional.normalize(vec_BD, dim=-1).detach()
            steered_BD = normalized_features * norms_B1 * steering_coefficient  # (B, d_model)
            # residual: (1, total_tokens, d_model)
            if not in_place:
                residual = residual.clone()
            residual[0, global_indices, :] = steered_BD.to(dtype)  # replace B locations
            if output_is_tuple:
                return (residual, *rest)
            else:
                return residual
        except Exception as e:
            breakpoint()
            raise e

    return hook_fn


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
