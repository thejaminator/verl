# %%
%load_ext autoreload
%autoreload 2


# %%
import os
os.environ['VLLM_USE_V1'] = '0'
os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"

import torch
import torch.nn as nn
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Callable
import contextlib
import numpy as np
from huggingface_hub import hf_hub_download
import gc


# %%
# Configuration matching host_vllm_server_hook.py
MODEL_NAME = "google/gemma-2-9b-it"
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
CTX_LEN = 512 * 4
LAYER = 9  # Target layer for activation steering
MAX_DECODE_TOKENS = 100

# SAE Configuration
SAE_REPO_ID = "google/gemma-scope-9b-it-res"
LOAD_LORAS = ["thejaminator/sae-introspection-lora"]
SAE_WIDTH = 16  # Can be 16 or 131
SAE_FILENAME = f"layer_{LAYER}/width_16k/average_l0_88/params.npz"
STEERING_COEFFICIENT = 2.0
SAE_INDEX = 10027  # Test with this SAE feature index

print(f"Model: {MODEL_NAME}")
print(f"Target layer: {LAYER}")
print(f"SAE index: {SAE_INDEX}")
print(f"Steering coefficient: {STEERING_COEFFICIENT}")

# %%
# SAE Classes (copied from host_vllm_server_hook.py)
class JumpReluSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.threshold = nn.Parameter(torch.zeros(d_sae, dtype=dtype, device=device))
        self.device = device
        self.dtype = dtype
        self.d_sae = d_sae
        self.d_in = d_in
        self.to(dtype=dtype, device=device)

    def encode(self, x: torch.Tensor):
        pre_acts = x @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, feature_acts: torch.Tensor):
        return feature_acts @ self.W_dec + self.b_dec


def load_gemma_scope_sae(
    repo_id: str,
    filename: str,
    layer: int,
    device: torch.device,
    dtype: torch.dtype,
    local_dir: str = "downloaded_saes",
) -> JumpReluSAE:
    """Load Gemma Scope SAE from Hugging Face Hub."""
    path_to_params = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_download=False,
        local_dir=local_dir,
    )
    pytorch_path = path_to_params.replace(".npz", ".pt")

    # Convert npz to pt for faster loading
    if not os.path.exists(pytorch_path):
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v) for k, v in params.items()}
        torch.save(pt_params, pytorch_path)

    pt_params = torch.load(pytorch_path)

    d_in = pt_params["W_enc"].shape[0]
    d_sae = pt_params["W_enc"].shape[1]

    sae = JumpReluSAE(d_in, d_sae, device, dtype)
    sae.load_state_dict(pt_params)
    sae.to(dtype=dtype, device=device)

    return sae


def get_sae_feature_vector(sae_index: int, sae: JumpReluSAE) -> torch.Tensor:
    """
    Get SAE feature vector for the given index from the loaded SAE.
    """
    if sae_index >= sae.d_sae:
        raise ValueError(f"Feature index {sae_index} is out of bounds for SAE with {sae.d_sae} features")

    # Use decoder weights as feature vectors (maps from feature space back to residual stream)
    return sae.W_dec[sae_index].clone()


def find_x_positions(formatted_prompt: str, tokenizer) -> list[int]:
    """Find positions of 'X' tokens in the formatted prompt."""
    positions = []
    tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    for i, token_id in enumerate(tokens):
        if tokenizer.decode([token_id]) == "X":
            positions.append(i)
    return positions

# %%
# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create test prompts with 'X' tokens
test_inputs = [
    [{"role": "user", "content": "Can you explain to me what 'X' means? Format your final answer with <explanation>"}],
    # [{"role": "user", "content": "What does the 'X' mean?"}]
]

# Apply chat template to both prompts
formatted_prompts = [
    tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    for prompt in test_inputs
]

print("Prompt 1:", formatted_prompts[0])
# print("Prompt 2:", formatted_prompts[1])

# Find positions of 'X' in each prompt
positions = []
for prompt in formatted_prompts:
    x_positions = find_x_positions(prompt, tokenizer)
    if x_positions:
        positions.append(x_positions[0])  # Take first 'X' if multiple
    else:
        raise ValueError(f"No 'X' token found in prompt: {prompt}")

print(f"X positions: {positions}")

# Tokenize both prompts with padding to handle different lengths
tokenized = tokenizer(formatted_prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(DEVICE)
print(f"Tokenized shape: {tokenized['input_ids'].shape}")
print(f"Attention mask shape: {tokenized['attention_mask'].shape}")

# %%
# Load SAE
print("Loading SAE...")
sae = load_gemma_scope_sae(
    repo_id=SAE_REPO_ID,
    filename=SAE_FILENAME,
    layer=LAYER,
    device=DEVICE,
    dtype=DTYPE,
)

print(f"SAE loaded - d_in: {sae.d_in}, d_sae: {sae.d_sae}")

# Get SAE feature vector for our test index
feature_vector = get_sae_feature_vector(SAE_INDEX, sae)
print(f"Feature vector shape: {feature_vector.shape}")
print(f"Feature vector norm: {feature_vector.norm().item():.4f}")

# Create feature vectors for both prompts (same vector for demo)
features = [feature_vector, feature_vector]

print(f"Number of prompts: {len(formatted_prompts)}")
print(f"Number of feature vectors: {len(features)}")
print(f"Number of positions: {len(positions)}")

# %%
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
    vectors: list[torch.Tensor],
    positions: list[int],
    steering_coefficient: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable:
    """
    Returns a forward hook that replaces specified residual-stream activations
    during the initial prompt pass of model.generate.
    
    vLLM version that works with flattened residual streams (B*L, d_model).
    
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
        resid_flat, *rest = output  # (B*L, d_model)
        total_tokens, _d_model = resid_flat.shape
        
        # Extract batch size and sequence length from the input
        # The actual sequence length should be inferred from total_tokens / B
        # since we trust B and know total_tokens
        L = total_tokens // B
        B_actual = B  # Use provided batch size

        # Only apply steering during prompt pass (sequence length > 1)
        if L <= 1:
            return (resid_flat, *rest)
        
        print(f"Applying feature vector on module {type(module).__name__}. Sequence length: {L}, Batch size: {B_actual}")
        
        # Reshape flattened tensor to (B_actual, L, d_model) format using actual tensor dimensions
        resid_BLD = resid_flat.view(B_actual, L, _d_model)
        
        # Safety: make sure every position is inside current sequence
        if (pos_B >= L).any():
            bad = pos_B[pos_B >= L].min().item()
            raise IndexError(f"position {bad} is out of bounds for length {L}")
        
        # ---- compute norms of original activations at the target slots ----
        batch_idx_B = torch.arange(B, device=device)  # (B,)
        orig_BD = resid_BLD[batch_idx_B, pos_B]  # (B, d_model)
        norms_B1 = orig_BD.norm(dim=-1, keepdim=True).detach()  # (B, 1)
        
        # ---- build steered vectors ----
        steered_BD = torch.nn.functional.normalize(vec_BD, dim=-1) * norms_B1 * steering_coefficient  # (B, d_model)
        
        # ---- in-place replacement via advanced indexing ----
        resid_BLD[batch_idx_B, pos_B] = steered_BD
        
        # Reshape back to flattened format (B*L, d_model)
        resid_flat_modified = resid_BLD.view(total_tokens, _d_model)
        
        return (resid_flat_modified, *rest)

    return hook_fn


# %%
# HF Transformers hook (copied from lightweight_sft.py)
def get_hf_activation_steering_hook(
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
    
    # Handle the case where positions might create a scalar tensor instead of a 1D tensor
    if pos_B.dim() == 0:  # scalar tensor case (when positions is a single integer instead of list)
        pos_B = pos_B.unsqueeze(0)  # Convert scalar to (1,) shape
    
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


def get_hf_submodule(model: AutoModelForCausalLM, layer: int, use_lora: bool = False):
    """Gets the residual stream submodule for HF transformers"""
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


def unload_model(model):
    """Properly unload a model to free GPU memory"""
    if hasattr(model, 'cpu'):
        model.cpu()
    del model
    torch.cuda.empty_cache()
    gc.collect()

# %%
# =============================================================================
# PART 1: HUGGING FACE TRANSFORMERS TEST
# =============================================================================
print("="*80)
print("PART 1: Testing with Hugging Face Transformers")
print("="*80)

# Load HF model with LoRA
print("Loading Hugging Face model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="auto", 
    torch_dtype=DTYPE, 
    attn_implementation="eager"
)

# Load LoRA adapter
print(f"Loading LoRA adapter: {LOAD_LORAS[0]}")
hf_model = PeftModel.from_pretrained(hf_model, LOAD_LORAS[0])

# Get the target submodule for the hook
hf_target_layer = get_hf_submodule(hf_model, LAYER, use_lora=True)

# Prepare the hook
hf_hook_fn = get_hf_activation_steering_hook(features, positions, STEERING_COEFFICIENT, DEVICE, DTYPE)

# Generate with LoRA + steering
print(f"Generating with HF Transformers (LoRA + Steering, SAE index {SAE_INDEX})...")
with add_hook(hf_target_layer, hf_hook_fn):
    with torch.no_grad():
        hf_outputs = hf_model.generate(
            **tokenized,
            do_sample=False,
            max_new_tokens=MAX_DECODE_TOKENS,
            pad_token_id=tokenizer.eos_token_id
        )

# Decode the new tokens only
hf_generated_tokens = hf_outputs[:, tokenized['input_ids'].shape[1]:]
hf_texts = tokenizer.batch_decode(hf_generated_tokens, skip_special_tokens=True)

print("HF Transformers outputs (LoRA + Steering):")
for i, text in enumerate(hf_texts):
    print(f"  Prompt {i+1}: {text}")

# Unload HF model to free memory
print("Unloading HF model...")
unload_model(hf_model)

# %%
# =============================================================================
# PART 2: VLLM TEST  
# =============================================================================
print("\n" + "="*80)
print("PART 2: Testing with vLLM")
print("="*80)

# Demo the activation steering hook with batch of prompts
batch_token_lists = tokenized["input_ids"].tolist()
sampling_params = SamplingParams(temperature=0.0, ignore_eos=False, max_tokens=MAX_DECODE_TOKENS)

# %%
# Initialize vLLM model with LoRA support
print("Initializing vLLM model...")
llm = LLM(
    model=MODEL_NAME,
    tensor_parallel_size=1,
    max_model_len=CTX_LEN,
    enforce_eager=True,
    dtype=DTYPE,
    disable_async_output_proc=True,
    gpu_memory_utilization=0.5,
    enable_lora=True,
    max_lora_rank=64,
)
vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

# Load LoRA adapters
if LOAD_LORAS:
    print(f"Loading LoRA adapters: {LOAD_LORAS}")
    for i, lora_id in enumerate(LOAD_LORAS, 1):
        print(f"Loading LoRA adapter: {lora_id}")
        lora_request = LoRARequest(lora_name=f"lora_{i}", lora_int_id=i, lora_path=lora_id)
        llm.llm_engine.add_lora(lora_request)

# %%
# Test with both LoRA and steering in vLLM
print(f"Generating with vLLM (LoRA + Steering, SAE index {SAE_INDEX})...")
vllm_hook_fn = get_activation_steering_hook(features, positions, STEERING_COEFFICIENT, DEVICE, DTYPE)

# Apply hook to the specified layer
vllm_target_layer = vllm_model.model.layers[LAYER]
lora_request = LoRARequest(lora_name="lora_1", lora_int_id=1, lora_path=LOAD_LORAS[0])

with add_hook(vllm_target_layer, vllm_hook_fn):
    vllm_lora_steered = llm.generate(
        prompt_token_ids=batch_token_lists, 
        sampling_params=sampling_params, 
        lora_request=lora_request,
        use_tqdm=False
    )

vllm_texts = [output.outputs[0].text for output in vllm_lora_steered]
print("vLLM outputs (LoRA + Steering):")
for i, text in enumerate(vllm_texts):
    print(f"  Prompt {i+1}: {text}")

# %%
# =============================================================================
# COMPARISON
# =============================================================================
print("\n" + "="*80)
print("COMPARISON: HF Transformers vs vLLM")
print("="*80)

for i in range(len(formatted_prompts)):
    print(f"\n--- Prompt {i+1} ---")
    print(f"Original prompt: {formatted_prompts[i]}")
    print(f"Target position for steering: {positions[i]}")
    print(f"HF Transformers (LoRA+Steering): {hf_texts[i]}")
    print(f"vLLM (LoRA+Steering):           {vllm_texts[i]}")
    print(f"Outputs match: {'YES' if hf_texts[i].strip() == vllm_texts[i].strip() else 'NO'}")
