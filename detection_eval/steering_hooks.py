from typing import Callable

import torch


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

            assert tokens_L[count : count + prompt_length].equal(expected_position_indices_L), (
                f"Position indices mismatch at index {count}, expected {expected_position_indices_L}, got {tokens_L[count : count + prompt_length]}"
            )

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
