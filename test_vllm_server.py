#!/usr/bin/env python3
"""
Test client for the vLLM server with activation steering hooks.
"""


import requests


def test_chat_completion(sae_index=None, steering_coefficient=2.0) -> str | None:
    """Test the chat completion endpoint."""

    url = "http://localhost:8000/v1/chat/completions"

    payload = {
        "messages": [{"role": "user", "content": "What is the meaning of the word 'X'?"}],
        "max_tokens": 100,
        "temperature": 0.0,
    }

    # Add steering parameters if provided
    if sae_index is not None:
        payload["sae_index"] = sae_index
        payload["steering_coefficient"] = steering_coefficient

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def main():
    """Run tests comparing baseline vs steered outputs."""

    print("Testing vLLM server with activation steering...")

    # Test baseline (no steering)
    print("\n=== Baseline (no steering) ===")
    baseline_output: str | None = test_chat_completion()
    if baseline_output:
        print(f"Output: {baseline_output}")

    # Test with different SAE indices
    for sae_index in [42, 123, 999]:
        print(f"\n=== With SAE index {sae_index} ===")
        steered_output = test_chat_completion(sae_index=sae_index, steering_coefficient=5.0)
        if steered_output:
            print(f"Output: {steered_output}")
            if baseline_output:
                print(f"Same as baseline: {'Yes' if steered_output == baseline_output else 'No'}")

    # Test health endpoint
    print("\n=== Health Check ===")
    health_response = requests.get("http://localhost:8000/health")
    if health_response.status_code == 200:
        print(f"Health: {health_response.json()}")

    # Test models endpoint
    print("\n=== Available Models ===")
    models_response = requests.get("http://localhost:8000/v1/models")
    if models_response.status_code == 200:
        models = models_response.json()
        print(f"Available models: {[model['id'] for model in models['data']]}")


if __name__ == "__main__":
    main()
