#!/usr/bin/env python3
"""
Test client for the vLLM server with activation steering hooks using OpenAI async client.
"""

import asyncio
from openai import AsyncOpenAI


async def test_chat_completion(
    client: AsyncOpenAI, 
    sae_index: int | None = None, 
) -> str | None:
    """Test the chat completion endpoint using OpenAI async client."""
    
    try:
        # Prepare extra parameters for steering
        extra_body = {}
        if sae_index is not None:
            extra_body["sae_index"] = sae_index
        
        response = await client.chat.completions.create(
            model="thejaminator/sae-introspection-lora",
            messages=[
                {"role": "user", "content": "Can you explain to me what 'X' means? Format your final answer with <explanation>"}
            ],
            max_tokens=500,
            temperature=0.0,
            extra_body=extra_body if extra_body else None
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None


async def test_health_check(client: AsyncOpenAI) -> None:
    """Test health endpoint using OpenAI client."""
    try:
        # Use the models endpoint as a health check
        models = await client.models.list()
        print(f"Available models: {[model.id for model in models.data]}")
    except Exception as e:
        print(f"Health check failed: {e}")


async def main():
    """Run tests comparing baseline vs steered outputs in parallel."""
    
    # Initialize OpenAI client pointed to local server
    client = AsyncOpenAI(
        api_key="dummy-key",  # vLLM doesn't require a real API key
        base_url="http://localhost:8000/v1"
    )

    print("Testing vLLM server with activation steering using OpenAI async client...")
    print("Running all requests in parallel...")

    # Create all tasks to run in parallel
    tasks = [
        # Baseline (no steering)
        test_chat_completion(client),
        # Tests with different SAE indices
        test_chat_completion(client, sae_index=42),
        test_chat_completion(client, sae_index=123),
        test_chat_completion(client, sae_index=999),

    ]

    # Run all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process and display results
    labels = [
        "Baseline (no steering)",
        "SAE index 42",
        "SAE index 123", 
        "SAE index 999",
     ]
    
    baseline_output = None
    
    for i, (label, result) in enumerate(zip(labels, results)):
        print(f"\n=== {label} ===")
        if isinstance(result, Exception):
            print(f"Error: {result}")

        else:
            if result:
                print(f"Output: {result}")
                if i == 0:  # Save baseline for comparison
                    baseline_output = result
                elif baseline_output and i > 0:
                    print(f"Same as baseline: {'Yes' if result == baseline_output else 'No'}")
            else:
                print("No output received")


if __name__ == "__main__":
    asyncio.run(main())
