#!/usr/bin/env python3
"""
Minimal SAE Feature Explanation Script

This script generates self-explanations for sparse autoencoder features using
activation steering with the Gemma-2-9B-IT model.
"""

import contextlib
import os
from abc import ABC, abstractmethod
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


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


def build_explanation_prompt(
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """
    Constructs a prompt for generating SAE feature explanations.

    Returns:
        A tuple containing the tokenized input IDs and the position of the 'X'
        placeholder where activations should be steered.
    """
    # Create chat format messages
    messages = [
        {
            "role": "user",
            "content": "Can you explain to me what 'X' means? Format your final answer with <explanation>",
        },
        {
            "role": "assistant",
            "content": "'x' means that",
        },
    ]

    # Apply chat template
    input_as_str: str = tokenizer.apply_chat_template(  # type: ignore
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    print(f"Formatted input: {input_as_str}")

    # Find the position of the placeholder 'X'
    token_ids = tokenizer.encode(input_as_str, add_special_tokens=False)
    print(f"Token IDs: {token_ids}")
    x_token_ids = tokenizer.encode("X", add_special_tokens=False)
    assert len(x_token_ids) == 1, "Expected to find 1 'X' token"
    x_token_id = x_token_ids[0]
    print(f"X token ID: {x_token_id}")
    positions = [i for i, token_id in enumerate(token_ids) if token_id == x_token_id]

    print(f"Found X token at position: {positions[0]}/{len(token_ids)}")

    # Debug: decode around the X position
    if positions:
        # Print a few tokens before and after the X token for context
        context_window = 3  # Number of tokens before and after X
        start_idx = max(0, positions[0] - context_window)
        end_idx = min(len(token_ids), positions[0] + context_window + 1)
        # 3 tokens before
        context_tokens = token_ids[start_idx : positions[0]]
        context_text = tokenizer.decode(context_tokens)
        print(f"Context before X: '{context_text}' (tokens {start_idx}-{positions[0] - 1})")
        # 3 tokens after
        context_tokens = token_ids[positions[0] + 1 : end_idx]
        context_text = tokenizer.decode(context_tokens)
        print(f"Context after X: '{context_text}' (tokens {positions[0] + 1}-{end_idx - 1})")

    assert len(positions) == 1, (
        f"Expected to find 1 'X' placeholder, but found {len(positions)}. Full prompt: {input_as_str}"
    )

    tokenized_input = tokenizer(str(input_as_str), return_tensors="pt", add_special_tokens=False).to(device)

    return tokenized_input.input_ids, positions[0]


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def nuclear_hook(module, _input, output):
    resid_BLD, *rest = output
    L = resid_BLD.shape[1]

    if L > 1:  # Only during initial prompt
        print("🚨 NUCLEAR HOOK: Zeroing ALL activations!")
        # Zero out everything - should completely break the model
        resid_BLD.zero_()

    return (resid_BLD, *rest)


def get_activation_steering_hook(
    # K = number of feature/steering vectors per batch item
    vectors: list[list[torch.Tensor]],  # [B][K, d_model]  or [K, d_model] if B==1
    positions: list[list[int]],  # [B][K]
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Returns a forward hook that *replaces* specified residual-stream activations
    during the initial prompt pass of `model.generate`.

    • vectors[b][k]  – feature vector to inject for batch b, slot k
    • positions[b][k]– token index (0-based, within prompt only)
    """

    # ---- pack Python lists → torch tensors once, outside the hook ----
    vec_BKD = torch.stack([torch.stack(v) for v in vectors])  # (B, K, d)
    pos_BK = torch.tensor(positions, dtype=torch.long)  # (B, K)

    B, K, d_model = vec_BKD.shape
    assert pos_BK.shape == (B, K)

    vec_BKD = vec_BKD.to(device, dtype)
    pos_BK = pos_BK.to(device)

    def hook_fn(module, _input, output):
        resid_BLD, *rest = output  # Gemma returns (resid, hidden_states, ...)
        L = resid_BLD.shape[1]

        # Only touch the *prompt* forward pass (sequence length > 1)
        if L <= 1:
            return (resid_BLD, *rest)

        print(
            f"Applying feature vector on module {type(module).__name__}. Sequence length: {L}, Batch size: {resid_BLD.shape[0]}"
        )

        # Safety: make sure every position is inside current sequence
        if (pos_BK >= L).any():
            bad = pos_BK[pos_BK >= L].min().item()
            raise IndexError(f"position {bad} is out of bounds for length {L}")

        # ---- compute norms of original activations at the target slots ----
        batch_idx_B1 = torch.arange(B, device=device).unsqueeze(1)  # (B, 1) → (B, K)
        orig_BKD = resid_BLD[batch_idx_B1, pos_BK]  # (B, K, d)
        norms_BK1 = orig_BKD.norm(dim=-1, keepdim=True)  # (B, K, 1)

        # ---- build steered vectors ----
        steered_BKD = torch.nn.functional.normalize(vec_BKD, dim=-1) * norms_BK1 * steering_coefficient  # (B, K, d)

        # ---- in-place replacement via advanced indexing ----
        resid_BLD[batch_idx_B1, pos_BK] = steered_BKD

        return (resid_BLD, *rest)

    return hook_fn


def main(
    sae_index: int = 0,
    steering_coefficient: float = 2.0,
    layer: int = 9,
    num_generations: int = 10,
):
    """
    Main function to generate SAE feature explanations.

    Args:
        sae_index: Index of the SAE feature to explain
        steering_coefficient: Strength of activation steering
        layer: Model layer to apply steering to
        num_generations: Number of explanations to generate
    """
    print(f"Generating {num_generations} explanations for SAE feature {sae_index}")
    print(f"Using steering coefficient: {steering_coefficient}, layer: {layer}")

    # Setup
    model_name = "google/gemma-2-9b-it"
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load SAE
    print(f"Loading SAE for layer {layer}...")
    sae_repo_id = "google/gemma-scope-9b-it-res"
    sae_filename = f"layer_{layer}/width_16k/average_l0_88/params.npz"

    sae = load_gemma_scope_jumprelu_sae(
        repo_id=sae_repo_id,
        filename=sae_filename,
        layer=layer,
        model_name=model_name,
        device=device,
        dtype=dtype,
    )

    # Get the model submodule for the specified layer
    submodule = get_submodule(model, layer)

    # Build prompt once
    orig_input_ids, x_position = build_explanation_prompt(tokenizer, device)
    orig_input_ids = orig_input_ids.squeeze()

    print(f"Original prompt length: {len(orig_input_ids)}")
    print(f"X position: {x_position}")
    print(f"Prompt: {tokenizer.decode(orig_input_ids)}")

    # Get feature vector (using decoder weights)
    feature_vector = sae.W_dec[sae_index]
    print(f"Feature vector shape: {feature_vector.shape}")

    # Prepare batch data for steering
    batch_steering_vectors = []
    batch_positions = []

    for i in range(num_generations):
        # Each batch item gets the same feature vector
        batch_steering_vectors.append([feature_vector])
        batch_positions.append([x_position])

    # Create batch input - repeat the same prompt for each generation
    input_ids_BL = einops.repeat(orig_input_ids, "L -> B L", B=num_generations)
    attn_mask_BL = torch.ones_like(input_ids_BL, dtype=torch.bool).to(device)

    tokenized_input = {
        "input_ids": input_ids_BL,
        "attention_mask": attn_mask_BL,
    }

    # Create steering hook
    hook_fn = get_activation_steering_hook(
        vectors=batch_steering_vectors,
        positions=batch_positions,
        steering_coefficient=steering_coefficient,
        device=device,
        dtype=dtype,
    )

    # Generation settings
    generation_kwargs = {
        "do_sample": True,
        "temperature": 1.0,
        "max_new_tokens": 200,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Generate all explanations at once
    print(f"\nGenerating {num_generations} explanations in batch...")
    print(f"Input shape: {tokenized_input['input_ids'].shape}")
    print(f"First few tokens: {tokenized_input['input_ids'][0, :10]}")

    # Try with a simple forward pass first (no generation)
    # with add_hook(submodule, hook_fn):
    #     # Just do a forward pass to see if steering works
    #     with torch.no_grad():
    #         print("Simple forward pass...")
    #         outputs = model(**tokenized_input)
    #         logits = outputs.logits
    #         next_token_logits = logits[:, -1, :]  # Last position
    #         next_tokens = torch.argmax(next_token_logits, dim=-1)
    #         print("Next tokens:", [tokenizer.decode(t) for t in next_tokens])

    with add_hook(submodule, hook_fn):
        output_ids = model.generate(**tokenized_input, **generation_kwargs)

    # Decode the generated tokens for each batch item
    explanations = []
    generated_tokens = output_ids[:, input_ids_BL.shape[1] :]

    for i in range(num_generations):
        decoded_output = tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
        explanations.append(decoded_output)
        print(f"\nGeneration {i + 1}/{num_generations}:")
        print(decoded_output)
        print("-" * 80)

    return explanations


if __name__ == "__main__":
    # Example usage
    explanations = main(
        sae_index=0,
        steering_coefficient=500.0,
        layer=9,
        num_generations=10,
    )

    print(f"\nGenerated {len(explanations)} explanations total.")
