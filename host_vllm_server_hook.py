#!/usr/bin/env python3
"""
FastAPI server that runs vLLM with activation steering hooks.
Mimics OpenAI chat completions API with additional sae_index parameter.
"""

import contextlib
import os
import time
import uuid
from typing import Any, Callable, Optional

import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Environment setup
os.environ["VLLM_USE_V1"] = "0"
os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"

# Configuration
MODEL_NAME = "google/gemma-2-2b-it"
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
CTX_LEN = 512
LAYER = 6  # Target layer for activation steering


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    model: Optional[str] = MODEL_NAME
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.0
    sae_index: Optional[int] = None  # Custom parameter for SAE feature index
    steering_coefficient: float = 2.0


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


class VLLMServer:
    """Encapsulates vLLM model and tokenizer state."""

    def __init__(self):
        self.llm: Optional[LLM] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.initialized = False

    def initialize(self):
        """Initialize the vLLM model and tokenizer."""
        if self.initialized:
            return

        print("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        print("Initializing vLLM model...")
        self.llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            max_model_len=CTX_LEN * 4,
            enforce_eager=True,
            dtype=DTYPE,
            disable_async_output_proc=True,
            gpu_memory_utilization=0.5,
        )
        self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        self.initialized = True
        print("Server ready!")

    def is_ready(self) -> bool:
        """Check if server is ready."""
        return self.initialized and self.llm is not None and self.tokenizer is not None


@contextlib.contextmanager
def add_hook(module: torch.nn.Module, hook: Callable):
    """Temporarily adds a forward hook to a model module."""
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
    """
    vec_BD = torch.stack(vectors)  # (B, d_model)
    pos_B = torch.tensor(positions, dtype=torch.long)  # (B,)

    B, d_model = vec_BD.shape
    assert pos_B.shape == (B,)

    vec_BD = vec_BD.to(device, dtype)
    pos_B = pos_B.to(device)

    def hook_fn(module, _input, output):
        resid_flat, *rest = output
        total_tokens, d_model = resid_flat.shape

        L = total_tokens // B

        # Only touch the prompt forward pass (sequence length > 1)
        if L <= 1:
            return (resid_flat, *rest)

        # Safety: make sure every position is inside current sequence
        if (pos_B >= L).any():
            bad = pos_B[pos_B >= L].min().item()
            raise IndexError(f"position {bad} is out of bounds for length {L}")

        # Compute flat indices for vLLM's layout
        batch_offsets = torch.arange(B, device=device) * L
        flat_indices = batch_offsets + pos_B

        # Compute norms of original activations at the target slots
        orig_BD = resid_flat[flat_indices]
        norms_B1 = orig_BD.norm(dim=-1, keepdim=True)

        # Build steered vectors
        steered_BD = torch.nn.functional.normalize(vec_BD, dim=-1) * norms_B1 * steering_coefficient

        # In-place replacement via flat indexing
        resid_flat[flat_indices] = steered_BD

        return (resid_flat, *rest)

    return hook_fn


def get_sae_feature_vector(sae_index: int) -> torch.Tensor:
    """
    Get SAE feature vector for the given index.
    For now, returns a random vector. In practice, this would load from actual SAE.
    """
    if MODEL_NAME == "google/gemma-2-2b-it":
        # Return random vector for demo (in practice, load from SAE)
        torch.manual_seed(sae_index)  # Deterministic based on index
        return torch.randn(2304)
    else:
        raise ValueError(f"Model {MODEL_NAME} not supported")


def find_x_positions(formatted_prompt: str, tokenizer) -> list[int]:
    """Find positions of 'X' tokens in the formatted prompt."""
    positions = []
    tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    for i, token_id in enumerate(tokens):
        if tokenizer.decode([token_id]) == "X":
            positions.append(i)
    return positions


# Create server instance and FastAPI app
server = VLLMServer()
app = FastAPI(title="vLLM Server with Activation Steering", version="1.0.0")


def get_server() -> VLLMServer:
    """Dependency to get server instance."""
    return server


@app.on_event("startup")
async def startup_event():
    """Initialize the vLLM model and tokenizer on startup."""
    server.initialize()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, server: VLLMServer = Depends(get_server)):
    """Create a chat completion with optional activation steering."""

    if not server.is_ready():
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Format the messages using chat template
    formatted_prompt = server.tokenizer.apply_chat_template(
        [msg.dict() for msg in request.messages], tokenize=False, add_generation_prompt=True
    )

    # Tokenize the prompt
    tokenized = server.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    token_list = tokenized["input_ids"].tolist()[0]

    # Set up sampling parameters
    sampling_params = SamplingParams(temperature=request.temperature, ignore_eos=False, max_tokens=request.max_tokens)

    # Generate response
    if request.sae_index is not None:
        # Find X positions for steering
        x_positions = find_x_positions(formatted_prompt, server.tokenizer)

        if not x_positions:
            raise HTTPException(status_code=400, detail="No 'X' token found in prompt for steering")

        # Get SAE feature vector
        feature_vector = get_sae_feature_vector(request.sae_index)

        # Create steering hook
        hook_fn = get_activation_steering_hook(
            vectors=[feature_vector],
            positions=[x_positions[0]],  # Use first X position
            steering_coefficient=request.steering_coefficient,
            device=DEVICE,
            dtype=DTYPE,
        )

        # Generate with steering
        target_layer = server.model.model.layers[LAYER]
        with add_hook(target_layer, hook_fn):
            outputs = server.llm.generate(
                prompt_token_ids=[token_list], sampling_params=sampling_params, use_tqdm=False
            )
    else:
        # Generate without steering
        outputs = server.llm.generate(prompt_token_ids=[token_list], sampling_params=sampling_params, use_tqdm=False)

    # Extract generated text
    generated_text = outputs[0].outputs[0].text

    # Create response
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model or MODEL_NAME,
        choices=[Choice(index=0, message=Message(role="assistant", content=generated_text), finish_reason="stop")],
        usage=Usage(
            prompt_tokens=len(token_list),
            completion_tokens=len(server.tokenizer.encode(generated_text, add_special_tokens=False)),
            total_tokens=len(token_list) + len(server.tokenizer.encode(generated_text, add_special_tokens=False)),
        ),
    )

    return response


@app.get("/health")
async def health_check(server: VLLMServer = Depends(get_server)):
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": server.is_ready()}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI API compatibility)."""
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local", "permission": []}]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
