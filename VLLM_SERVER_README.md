# vLLM Server with Activation Steering

A FastAPI server that runs vLLM with activation steering hooks, mimicking the OpenAI chat completions API with an additional `sae_index` parameter for feature steering.

## Features

- **OpenAI-compatible API**: Mimics `/v1/chat/completions` endpoint
- **Activation Steering**: Inject SAE features via `sae_index` parameter
- **Single Generation**: Processes one request at a time (as requested)
- **Hardcoded Model**: Uses `google/gemma-2-2b-it` on layer 6
- **Minimal Error Handling**: Clean code without excessive try-except blocks

## Installation

```bash
pip install fastapi uvicorn vllm transformers torch
```

## Usage

### 1. Start the Server

```bash
python host_vllm_server_hook.py
```

The server will start on `http://localhost:8000` and initialize the Gemma-2-2b-it model.

### 2. Make Requests

#### Without Steering (Baseline)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the meaning of the word '\''X'\''?"}],
    "max_tokens": 100,
    "temperature": 0.0
  }'
```

#### With Activation Steering
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the meaning of the word '\''X'\''?"}],
    "max_tokens": 100,
    "temperature": 0.0,
    "sae_index": 42,
    "steering_coefficient": 5.0
  }'
```

### 3. Test with Python Client

```bash
python test_vllm_server.py
```

## API Parameters

### Standard OpenAI Parameters
- `messages`: List of conversation messages
- `model`: Model name (optional, defaults to gemma-2-2b-it)
- `max_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature

### Custom Steering Parameters
- `sae_index`: SAE feature index to inject (optional, enables steering when provided)
- `steering_coefficient`: Strength of steering (default: 5.0)

## How It Works

1. **Prompt Processing**: Applies chat template and tokenizes input
2. **Position Finding**: Locates 'X' tokens in the prompt for steering
3. **Feature Loading**: Gets SAE feature vector based on `sae_index` (currently random for demo)
4. **Hook Application**: Applies activation steering hook to layer 6 during generation
5. **Response**: Returns OpenAI-compatible response with generated text

## Endpoints

- `POST /v1/chat/completions` - Main chat completion endpoint
- `GET /health` - Health check
- `GET /v1/models` - List available models (OpenAI compatibility)

## Configuration

Edit the constants in `host_vllm_server_hook.py`:
- `MODEL_NAME`: Model to use
- `LAYER`: Target layer for steering
- `CTX_LEN`: Context length
- `DTYPE`: Model dtype

## Notes

- Currently uses random feature vectors for demo purposes
- In production, replace `get_sae_feature_vector()` with actual SAE loading
- Server processes one generation at a time as requested
- Requires 'X' token in prompt for steering to work
