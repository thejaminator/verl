# %%
%load_ext autoreload
%autoreload 2


# %%
import os
os.environ['VLLM_USE_V1'] = '0'
os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import Callable
import contextlib

import interp_tools.model_utils as model_utils


# %%
model_name = "google/gemma-2-2b-it"
dtype = torch.bfloat16
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)
ctx_len = 512
max_decode_tokens = 100

# %%

test_input = [{"role": "user", "content": "What is the meaning of the word 'X'?"}]

test_input = tokenizer.apply_chat_template(
    test_input, tokenize=False, add_generation_prompt=True
)

print(test_input)

positions = [
    i
    for i, a in enumerate(tokenizer.encode(test_input, add_special_tokens=False))
    if tokenizer.decode([a]) == "X"
]

orig_input = tokenizer(test_input, return_tensors="pt", add_special_tokens=False).to(
    device
)

# %%
layer = 6

# Create a random feature vector for demo (in practice, this would be from SAE)
if model_name == "google/gemma-2-2b-it":
    feature = torch.randn(2304)
else:
    raise ValueError(f"Model {model_name} not supported")

print(f"Feature vector shape: {feature.shape}")



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


def get_activation_steering_hook(
    vectors: list[torch.Tensor],  # [B] each with shape [d_model]
    positions: list[int],  # [B]
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Returns a forward hook that *replaces* specified residual-stream activations
    during the initial prompt pass of `model.generate`.

    • vectors[b]   – feature vector to inject for batch b
    • positions[b] – token index (0-based, within prompt only)
    """

    # ---- pack Python lists → torch tensors once, outside the hook ----
    vec_BD = torch.stack(vectors)  # (B, d_model)
    pos_B = torch.tensor(positions, dtype=torch.long)  # (B,)

    B, d_model = vec_BD.shape
    assert pos_B.shape == (B,)

    vec_BD = vec_BD.to(device, dtype)
    pos_B = pos_B.to(device)

    def hook_fn(module, _input, output):
        print(f"🔥 HOOK CALLED! Module: {type(module).__name__}")
        resid_flat, *rest = output  # vLLM uses flattened layout: (batch_size * seq_len, d_model)
        total_tokens, d_model = resid_flat.shape
        print(f"🔥 Hook processing: total_tokens {total_tokens}, d_model {d_model}, shape {resid_flat.shape}")

        # vLLM flattens (B, L) -> (B*L), so we need to infer L from total_tokens
        # Assuming all sequences have the same length L
        L = total_tokens // B
        print(f"🔥 Inferred: batch_size={B}, seq_len={L}")

        # Only touch the *prompt* forward pass (sequence length > 1)
        if L <= 1:
            print(f"Skipping hook because sequence length is <= 1, total_tokens: {total_tokens}")
            return (resid_flat, *rest)

        print(f"Applying feature vector on module {type(module).__name__}. Sequence length: {L}, Batch size: {B}")

        # Safety: make sure every position is inside current sequence
        if (pos_B >= L).any():
            bad = pos_B[pos_B >= L].min().item()
            raise IndexError(f"position {bad} is out of bounds for length {L}")

        # ---- compute flat indices for vLLM's layout ----
        # For batch b at position p: flat_index = b * L + p
        batch_offsets = torch.arange(B, device=device) * L  # (B,)
        flat_indices = batch_offsets + pos_B  # (B,)

        # ---- compute norms of original activations at the target slots ----
        orig_BD = resid_flat[flat_indices]  # (B, d_model)
        norms_B1 = orig_BD.norm(dim=-1, keepdim=True)  # (B, 1)

        # ---- build steered vectors ----
        steered_BD = torch.nn.functional.normalize(vec_BD, dim=-1) * norms_B1 * steering_coefficient  # (B, d_model)

        # ---- in-place replacement via flat indexing ----
        resid_flat[flat_indices] = steered_BD

        return (resid_flat, *rest)

    return hook_fn


# %%
# Initialize vLLM model

llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    max_model_len=ctx_len*4,
    enforce_eager=True,
    dtype=dtype,
    disable_async_output_proc=True,
    gpu_memory_utilization=0.5,
)
model = llm.llm_engine.model_executor.driver_worker.model_runner.model

# %%
# Demo the activation steering hook
list_tokens = orig_input["input_ids"].tolist()[0]
sampling_params = SamplingParams(temperature=0.0, ignore_eos=False, max_tokens=max_decode_tokens)

# Generate baseline output without steering
print("Generating baseline output...")
vllm_baseline = llm.generate(prompt_token_ids=list_tokens, sampling_params=sampling_params, use_tqdm=False)
baseline_text = vllm_baseline[0].outputs[0].text
print(f"Baseline output: {baseline_text}")

# %%
# Now generate with activation steering
print("\nGenerating with activation steering...")
hook_fn = get_activation_steering_hook([feature], positions, 5.0, device, dtype)

# Apply hook to the specified layer
target_layer = model.model.layers[layer]
with add_hook(target_layer, hook_fn):
    vllm_steered = llm.generate(prompt_token_ids=list_tokens, sampling_params=sampling_params, use_tqdm=False)

steered_text = vllm_steered[0].outputs[0].text
print(f"Steered output: {steered_text}")

# %%
# Compare outputs
print("\n=== COMPARISON ===")
print(f"Original prompt: {test_input}")
print(f"Target positions for steering: {positions}")
print(f"Baseline output: {baseline_text}")
print(f"Steered output:  {steered_text}")
print(f"\nOutputs are {'SAME' if baseline_text == steered_text else 'DIFFERENT'}")

