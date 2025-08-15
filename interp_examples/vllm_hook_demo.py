# %%
%load_ext autoreload
%autoreload 2


# %%
import os
os.environ['VLLM_USE_V1'] = '0'
os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"

import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.logits_processor import _prune_hidden_states
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from jaxtyping import Float
from typing import Callable
import contextlib
from torch import Tensor

import interp_tools.model_utils as model_utils
import interp_tools.saes.jumprelu_sae as jumprelu_sae


# %%
model_name = "google/gemma-2-2b-it"
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
dtype = torch.bfloat16
device = torch.device("cuda")
dataset_name = "lmsys/lmsys-chat-1m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ctx_len = 512

max_decode_tokens = 100
n_samples = 100

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, ignore_eos=True, max_tokens=max_decode_tokens + 1)


generation_kwargs = {
    "max_new_tokens": max_decode_tokens+1,
    "do_sample": False,
}

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

hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=dtype)

# %%
with torch.no_grad():
    hf_output_orig = hf_model.generate(**orig_input, **generation_kwargs, output_logits=True, return_dict_in_generate=True)
    logits_hf_orig = torch.stack(hf_output_orig.logits)

# %%

print(hf_model)

# %%
layer = 20
layer = 6

repo_id = "google/gemma-scope-2b-pt-res"
filename = f"layer_{layer}/width_16k/average_l0_71/params.npz"
filename = f"layer_6/width_16k/average_l0_70/params.npz"
# repo_id = "google/gemma-scope-9b-it-res"
# layer = 9
# filename = f"layer_9/width_16k/average_l0_88/params.npz"
# model_name = "google/gemma-2-9b-it"

# sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
#     repo_id, filename, layer, model_name, device, dtype
# )
# %%
feature_idx = 0
# feature = sae.W_enc[:, feature_idx]
# feature = sae.W_dec[feature_idx]

if model_name == "meta-llama/Llama-3.1-8B-Instruct":
    feature = torch.randn(4096)
elif model_name == "google/gemma-2-2b-it":
    feature = torch.randn(2304)
else:
    raise ValueError(f"Model {model_name} not supported")

print(feature.shape)

submodule = model_utils.get_submodule(hf_model, layer)



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


def get_activation_addition_output_hook(
    vectors: list[Float[Tensor, "d_model"]],
    coeffs: list[float],
    positions: list[int],
) -> Callable:
    """Creates a hook function that adds scaled vectors to layer activations.

    This hook performs a simple activation steering by adding scaled vectors
    to the layer's output activations. This is the most basic form of intervention.

    Args:
        vectors: List of vectors to add, each of shape (d_model,)
        coeffs: List of scaling coefficients for each vector

    Returns:
        Hook function that modifies layer activations

    """

    assert len(vectors) == len(coeffs) == len(positions), (
        f"len(vectors): {len(vectors)}, len(coeffs): {len(coeffs)}, len(positions): {len(positions)}"
    )

    def hook_fn(module, input, output):
        resid_BLD, *rest = output if isinstance(output, tuple) else (output,)
        resid_BLD = resid_BLD.clone()

        if resid_BLD.shape[1] > 1:
            for i, (vector, coeff, position) in enumerate(
                zip(vectors, coeffs, positions)
            ):
                i = 0  # TODO: Better solution
                vector = vector.to(resid_BLD.device)
                # resid_vector = resid_BLD[i, position]
                resid_norm = torch.norm(resid_BLD[i, position])

                # resid_vector = resid_BLD[i, position]
                # resid_norm = torch.norm(resid_vector)
                # vector = vector / torch.norm(vector) * resid_norm * 1
                # resid_BLD[i, position] = vector

                normalized_vector = vector / torch.norm(vector)
                # normalized_resid_vector = resid_vector / torch.norm(resid_vector)

                # Mix half original resid_vector and half normalized vector
                # mixed_vector = 0.0 * normalized_resid_vector + 1.0 * normalized_vector
                mixed_vector = 1.0 * normalized_vector
                # Normalize to maintain original norm
                mixed_vector = mixed_vector * resid_norm
                mixed_vector = mixed_vector * coeff

                resid_BLD[i, position] = mixed_vector

        return (resid_BLD, *rest)

    return hook_fn


# %%

hf_acts_BLD = model_utils.collect_activations(hf_model, submodule, orig_input)
print(hf_acts_BLD.shape)






# %%

hook_fn = get_activation_addition_output_hook([feature], [5.0], positions)

context_manager = add_hook(submodule, hook_fn)

with context_manager:
    output_hf_v2= hf_model.generate(**orig_input, **generation_kwargs, output_logits=True, return_dict_in_generate=True)

logits_hf_v2 = torch.stack(output_hf_v2.logits)

# %%

diff = logits_hf_v2 - logits_hf_orig
print(f"mean diff: {diff.abs().mean()}, max diff: {diff.abs().max()}")
# print(diff.abs().sum(dim=-1))

# %%

# llm_8bit = LLM(
#     model=model_name,
#     tensor_parallel_size=1,
#     max_model_len=ctx_len*2,
#     enforce_eager=True,
#     dtype=dtype,
#     disable_async_output_proc=True,
#     quantization="fp8",
# )

llm = LLM(
    model=model_name,
    tensor_parallel_size=1,
    max_model_len=ctx_len*8,
    enforce_eager=True,
    dtype=dtype,
    disable_async_output_proc=True,
    gpu_memory_utilization=0.5,
)
model = llm.llm_engine.model_executor.driver_worker.model_runner.model

# %%

list_tokens = orig_input["input_ids"].tolist()[0]
sampling_params = SamplingParams(temperature=0.0, ignore_eos=False, max_tokens=max_decode_tokens+1)

logits_processor = llm.llm_engine.model_executor.driver_worker.model_runner.model.logits_processor
saved_logits = []
def logits_saving_hook(module, input, output):
    saved_logits.append(output[0].detach().clone())

saved_logits_handle = logits_processor.register_forward_hook(logits_saving_hook)

try:
    vllm_orig = llm.generate(prompt_token_ids=list_tokens, sampling_params=sampling_params, use_tqdm=False)
except Exception as e:
    print(e)
finally:
    saved_logits_handle.remove()


# %%

print(len(saved_logits))
print(len(saved_logits[0]))

vllm_orig_logits = torch.stack(saved_logits)

print(vllm_orig_logits.shape)
print(logits_hf_orig.squeeze(1).shape)


vllm_output_tokens = vllm_orig[0].outputs[0].token_ids
hf_output_tokens = hf_output_orig.sequences[0].tolist()


input_length = len(orig_input["input_ids"][0])
hf_output_tokens = hf_output_tokens[input_length:]

print(vllm_output_tokens)
print(hf_output_tokens)
# Find where tokens first diverge
diverge_idx = None
min_len = min(len(vllm_output_tokens), len(hf_output_tokens))
for i in range(min_len):
    if vllm_output_tokens[i] != hf_output_tokens[i]:
        diverge_idx = i
        break

print(f"Tokens diverge at index: {diverge_idx}")
if diverge_idx is not None:
    print(f"vLLM token at divergence: {vllm_output_tokens[diverge_idx]}")
    print(f"HF token at divergence: {hf_output_tokens[diverge_idx]}")
else:
    diverge_idx = len(vllm_output_tokens) - 1


idx = diverge_idx - 1

diff = vllm_orig_logits[:idx] - logits_hf_orig.squeeze(1)[:idx]
print(diff.shape)
print(f"mean diff: {diff.abs().mean()}, max diff: {diff.abs().max()}")

# print(vllm_output_tokens)

# print(vllm_output_tokens[0].outputs[0].text)
# print(tokenizer.decode(hf_output_orig.sequences[0]))
# print(hf_output_orig.sequences)

# %%

temp_saved_activations = []
tokens_len = len(list_tokens)
def activation_saving_hook(module, input, output):
    if output[0].shape[0] > 1:

        # assert input[0][tokens_len-1] == tokens_len-1, f"Seq length mismatch: {input[0][tokens_len-1]} != {tokens_len-1}"

        print(len(input), input[0].shape, input[1].shape, output[0].shape, output[1].shape)
        if len(input) == 3:
            print(input[0][:])
        temp_saved_activations.append(output[1].detach().clone())

        # temp_saved_activations.append(input[2].detach().clone())


saved_activations_handle = model.model.layers[layer].register_forward_hook(activation_saving_hook)

try:
    vllm_new = llm.generate(prompt_token_ids=list_tokens, sampling_params=sampling_params, use_tqdm=False)
except Exception as e:
    print(e)
finally:
    saved_activations_handle.remove()

print(vllm_new[0].outputs[0].token_ids)
print(temp_saved_activations[0].shape)

# %%

# TODO: PROBLEM HERE: numerical correctness in gemma-2-2b-it: 
# mean diff: 0.64453125, max diff: 164.0

hf_acts_BLD = hf_acts_BLD.squeeze(0)
vllm_acts_BLD = temp_saved_activations[0]

print(hf_acts_BLD.shape)
print(vllm_acts_BLD.shape)

print(hf_acts_BLD.abs().mean(), hf_acts_BLD.abs().max())
print(vllm_acts_BLD.abs().mean(), vllm_acts_BLD.abs().max())

diff = vllm_acts_BLD - hf_acts_BLD
print(diff.shape)
print(f"mean diff: {diff.abs().mean()}, max diff: {diff.abs().max()}")

# %%





# %%

vllm_new = llm.generate(prompt_token_ids=list_tokens, sampling_params=sampling_params, use_tqdm=False)

print(vllm_new[0].outputs[0].token_ids)

print(len(vllm_new))
print(len(vllm_new[0].outputs))

# %%


def get_vllm_activation_addition_output_hook(
    vectors: list[Float[Tensor, "d_model"]],
    coeffs: list[float],
    positions: list[int],
    seq_lengths: list[int],
) -> Callable:
    """Creates a hook function that adds scaled vectors to layer activations.

    This hook performs a simple activation steering by adding scaled vectors
    to the layer's output activations. This is the most basic form of intervention.

    Args:
        vectors: List of vectors to add, each of shape (d_model,)
        coeffs: List of scaling coefficients for each vector

    Returns:
        Hook function that modifies layer activations

    """

    assert len(vectors) == len(coeffs) == len(positions), (
        f"len(vectors): {len(vectors)}, len(coeffs): {len(coeffs)}, len(positions): {len(positions)}"
    )

    def hook_fn(module, input, output):
        resid_BLD, *rest = output if isinstance(output, tuple) else (output,)
        resid_BLD = resid_BLD.clone()

        if resid_BLD.shape[1] > 1:
            for i, (vector, coeff, position) in enumerate(
                zip(vectors, coeffs, positions)
            ):
                i = 0  # TODO: Better solution
                vector = vector.to(resid_BLD.device)
                resid_norm = torch.norm(resid_BLD[i, position])

                mixed_vector = vector / torch.norm(vector)
                mixed_vector = mixed_vector * resid_norm
                mixed_vector = mixed_vector * coeff

                resid_BLD[i, position] = mixed_vector

        return (resid_BLD, *rest)

    return hook_fn



list_tokens = orig_input["input_ids"].tolist()[0]
sampling_params = SamplingParams(temperature=0.0, ignore_eos=False, max_tokens=max_decode_tokens+1)

logits_processor = llm.llm_engine.model_executor.driver_worker.model_runner.model.logits_processor
saved_logits = []
def logits_saving_hook(module, input, output):
    saved_logits.append(output[0].detach().clone())

saved_logits_handle = logits_processor.register_forward_hook(logits_saving_hook)

try:
    vllm_orig = llm.generate(prompt_token_ids=list_tokens, sampling_params=sampling_params, use_tqdm=False)
except Exception as e:
    print(e)
finally:
    saved_logits_handle.remove()

# %%

