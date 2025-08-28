#!/usr/bin/env python3
"""
FastAPI server that runs vLLM with activation steering hooks.
Mimics OpenAI chat completions API with additional sae_index parameter.
"""

import asyncio
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from detection_eval.steering_hooks import add_hook, get_vllm_steering_hook

# Environment setup
os.environ["VLLM_USE_V1"] = "0"
os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"

# Configuration
MODEL_NAME = "google/gemma-2-9b-it"
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
CTX_LEN = 2000
SAE_LAYER = 9  # Target layer for activation steering (matching lightweight_sft.py)
GENERATE_WAIT_SECONDS = 2

# SAE Configuration
SAE_REPO_ID = "google/gemma-scope-9b-it-res"
load_loras = [
    # "thejaminator/sae-introspection-lora",
    # # 1000 steps
    # "thejaminator/gemma-introspection-20250821-step-250",
    # # 2000 steps
    # "thejaminator/gemma-introspection-20250821-step-500",
    # # 4000 steps
    # "thejaminator/gemma-introspection-20250821-step-1000",
    # 8000 steps
    "thejaminator/gemma-introspection-20250821",
    "thejaminator/gemma-hook-layer-0",
    # "thejaminator/gemma-25aug-22k",
    # "thejaminator/gemma-feelings-step-4000",
    # "thejaminator/gemma-feelings",
    # "thejaminator/gemma-retry",
    # "thejaminator/gemma-multiepoch",
    # "thejaminator/gemma-posneg-cot",
]
SAE_WIDTH = 131  # Can be 16 or 131. Check what we trained with?
SAE_FILENAME = f"layer_{SAE_LAYER}/width_131k/average_l0_121/params.npz"
STEERING_COEFFICIENT = 2.0
# INFO 08-21 04:36:17 [executor_base.py:118] Maximum concurrency for 2000 tokens per request: 55.05x
gpu_memory_utilization = 0.6
# Max batch size that we call .generate with. See vllm logs for the max it can take.
# IMPORTANT SHOULD BE >20%? THAN THE MAX PARALLELISM THAT VLLM WILL RUN.
# OTHERWISE VLLM WILL DO WEIRD THINGS THAT MESSS UP THE HOOK?
MAX_PARALLEL_REQUESTS = 36


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    model: str = MODEL_NAME
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.0
    sae_index: Optional[int] = None  # Custom parameter for SAE feature index
    hook_onto_layer: int = 9  # Default to 9 for historical reasons


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


@dataclass(kw_only=True)
class QueuedRequest:
    request: ChatCompletionRequest
    request_id: str
    timestamp: float
    future: asyncio.Future  # asyncio.Future for response


class VLLMServer:
    """Encapsulates vLLM model, tokenizer, SAE state, and request queues."""

    def __init__(self):
        print("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        print("Initializing vLLM model...")
        self.llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            max_model_len=CTX_LEN,
            dtype=DTYPE,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=64,
            disable_async_output_proc=True,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
        self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

        # Load LoRA adapters
        if load_loras:
            print(f"Loading LoRA adapters: {load_loras}")
            for i, lora_id in enumerate(load_loras, 1):
                print(f"Loading LoRA adapter: {lora_id}")
                lora_request = LoRARequest(lora_name=f"lora_{i}", lora_int_id=i, lora_path=lora_id)
                self.llm.llm_engine.add_lora(lora_request)

        print("Loading SAE...")
        self.sae = load_gemma_scope_sae(
            repo_id=SAE_REPO_ID,
            filename=SAE_FILENAME,
            layer=SAE_LAYER,
            device=DEVICE,
            dtype=DTYPE,
        )

        self.initialized = True
        print("Server ready!")

        # Queue management - separate queue per LoRA model
        self.queues: dict[str, deque[QueuedRequest]] = defaultdict(deque)
        self.processing_lock = asyncio.Lock()  # Ensure synchronous generation

    def get_model_key(self, model_name: str, hook_onto_layer: int) -> str:
        """Get the key for queue management based on model name."""
        if model_name == MODEL_NAME:
            return "base"
        return model_name + str(hook_onto_layer)

    async def add_to_queue(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Add request to appropriate queue and handle processing."""
        model_key = self.get_model_key(request.model, hook_onto_layer=request.hook_onto_layer)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Create future for async response
        future = asyncio.Future()

        queued_request = QueuedRequest(request=request, request_id=request_id, timestamp=time.time(), future=future)

        # Add to queue
        self.queues[model_key].append(queued_request)

        # Wait for response
        return await future

    async def process_queue(self, model_key: str):
        """Process up to MAX_PARALLEL_REQUESTS from the queue for a specific model."""
        async with self.processing_lock:  # Ensure only one batch processes at a time
            if not self.queues[model_key]:
                return

            # Get up to MAX_PARALLEL_REQUESTS current requests from queue
            batch_requests = []
            num_to_process = min(MAX_PARALLEL_REQUESTS, len(self.queues[model_key]))
            for _ in range(num_to_process):
                batch_requests.append(self.queues[model_key].popleft())

            # Process batch synchronously
            await self._process_batch(batch_requests, model_key)

    def _should_process_now(self, model_key: str) -> bool:
        """Return True if this queue should be processed now based on size or wait time."""
        queue = self.queues[model_key]
        if not queue:
            return False
        if len(queue) >= MAX_PARALLEL_REQUESTS:
            return True
        oldest_timestamp = queue[0].timestamp
        return (time.time() - oldest_timestamp) >= GENERATE_WAIT_SECONDS

    async def scheduler_loop(self) -> None:
        """Background loop that checks queues and processes them when ready."""
        while True:
            for model_key in list(self.queues.keys()):
                if self._should_process_now(model_key):
                    await self.process_queue(model_key)

            await asyncio.sleep(0.05)

    async def _process_batch(self, batch_requests: list[QueuedRequest], model_key: str):
        try:
            # Prepare batch data
            prompts = []
            token_lists = []
            steering_vectors = []
            steering_positions = []

            # Determine LoRA request (all requests in batch should use same model)
            lora_request: LoRARequest | None = None
            model_name = batch_requests[0].request.model
            if model_name != MODEL_NAME:
                for i, lora_id in enumerate(load_loras, 1):
                    if model_name == lora_id or model_name == f"lora_{i}":
                        lora_request = LoRARequest(lora_name=f"lora_{i}", lora_int_id=i, lora_path=lora_id)
                        break
                if lora_request is None:
                    raise HTTPException(status_code=400, detail=f"Model {model_name} not found in loaded LoRAs")

            # Process each request in the batch
            for queued_request in batch_requests:
                request = queued_request.request

                # Format prompt
                formatted_prompt = self.tokenizer.apply_chat_template(
                    [msg.dict() for msg in request.messages], tokenize=False, add_generation_prompt=True
                )
                prompts.append(formatted_prompt)

                # Tokenize
                tokenized = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
                token_list = tokenized["input_ids"].tolist()[0]
                token_lists.append(token_list)

                # Handle steering
                if request.sae_index is not None:
                    # Find X positions for steering
                    x_positions = find_x_positions(formatted_prompt, self.tokenizer)
                    print(f"X positions: {x_positions}")
                    if not x_positions:
                        raise HTTPException(status_code=400, detail="No 'X' token found in prompt for steering")

                    # Get SAE feature vector
                    feature_vector = get_sae_feature_vector(request.sae_index, self.sae)
                    print(f"Feature vector: {feature_vector}")
                    steering_vectors.append(feature_vector)
                    steering_positions.append(x_positions[0])
                else:
                    # Use zero vector for non-steering requests
                    zero_vector = torch.zeros(self.sae.d_in, dtype=DTYPE, device=DEVICE)
                    steering_vectors.append(zero_vector)
                    steering_positions.append(0)  # Position 0 as fallback

            temperature = batch_requests[0].request.temperature
            max_tokens = batch_requests[0].request.max_tokens
            sampling_params = [
                SamplingParams(temperature=temperature, ignore_eos=False, max_tokens=max_tokens)
                for _ in range(len(batch_requests))
            ]
            prompt_lengths = [len(token_list) for token_list in token_lists]

            # Create steering hook for the entire batch
            hook_fn = get_vllm_steering_hook(
                vectors=steering_vectors,
                positions=steering_positions,
                prompt_lengths=prompt_lengths,
                steering_coefficient=STEERING_COEFFICIENT,
                device=DEVICE,
                dtype=DTYPE,
            )
            # get the layer number from the hook_onto_layer
            layer_number = batch_requests[0].request.hook_onto_layer

            # Generate batch with steering
            target_layer = self.model.model.layers[layer_number]
            with torch.no_grad():
                with add_hook(target_layer, hook_fn):
                    outputs = self.llm.generate(
                        prompt_token_ids=token_lists,
                        sampling_params=sampling_params,
                        lora_request=lora_request,
                        use_tqdm=False,
                    )

            # Process outputs and set results - FIXED VERSION
            assert len(outputs) == len(batch_requests), (
                f"Output count {len(outputs)} doesn't match request count {len(batch_requests)}"
            )

            for queued_request, output in zip(batch_requests, outputs, strict=True):
                try:
                    generated_text = output.outputs[0].text

                    response = ChatCompletionResponse(
                        id=queued_request.request_id,  # ← Fixed: use the actual request_id
                        created=int(time.time()),
                        model=queued_request.request.model or MODEL_NAME,
                        choices=[
                            Choice(
                                index=0, message=Message(role="assistant", content=generated_text), finish_reason="stop"
                            )
                        ],
                        usage=Usage(
                            prompt_tokens=len(
                                self.tokenizer.encode(
                                    prompts[batch_requests.index(queued_request)], add_special_tokens=False
                                )
                            ),
                            completion_tokens=len(self.tokenizer.encode(generated_text, add_special_tokens=False)),
                            total_tokens=len(
                                self.tokenizer.encode(
                                    prompts[batch_requests.index(queued_request)], add_special_tokens=False
                                )
                            )
                            + len(self.tokenizer.encode(generated_text, add_special_tokens=False)),
                        ),
                    )

                    queued_request.future.set_result(response)

                except Exception as e:
                    # Handle individual request failure
                    print(f"Error processing individual request {queued_request.request_id}: {e}")
                    queued_request.future.set_exception(e)

        except Exception as e:
            # If batch processing fails completely, set exception for all unprocessed requests
            print(f"Batch processing failed: {e}")
            for queued_request in batch_requests:
                if not queued_request.future.done():
                    queued_request.future.set_exception(e)


# SAE Classes
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


# Create and initialize server instance immediately
server = VLLMServer()

app = FastAPI(title="vLLM Server with Activation Steering", version="1.0.0")


def get_server() -> VLLMServer:
    """Dependency to get server instance."""
    return server


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, server: VLLMServer = Depends(get_server)):
    """Create a chat completion with optional activation steering using queue system."""
    print(f"Received request: {request}")

    # Add request to queue and wait for response
    return await server.add_to_queue(request)


@app.get("/health")
async def health_check(server: VLLMServer = Depends(get_server)):
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": True}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI API compatibility)."""
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local", "permission": []}]}


@app.get("/queue/status")
async def queue_status(server: VLLMServer = Depends(get_server)):
    """Get current queue status for all models."""
    status = {}
    for model_key, queue in server.queues.items():
        status[model_key] = {
            "queue_length": len(queue),
            "oldest_timestamp": queue[0].timestamp if queue else None,
            "waiting_time": time.time() - queue[0].timestamp if queue else 0,
        }
    return {
        "queue_status": status,
        "max_parallel_requests": MAX_PARALLEL_REQUESTS,
        "generate_wait_seconds": GENERATE_WAIT_SECONDS,
    }


@app.on_event("startup")
async def start_scheduler():
    # Launch background scheduler loop
    asyncio.create_task(server.scheduler_loop())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
