#!/usr/bin/env python3
"""
Test client for the vLLM server with activation steering hooks using OpenAI async client.
"""

import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel
from slist import Slist


class TestChatCompletion(BaseModel):
    sae_index: int | None = None
    response: str


async def test_chat_completion(
    client: AsyncOpenAI,
    sae_index: int | None = None,
) -> TestChatCompletion | None:
    """Test the chat completion endpoint using OpenAI async client."""

    # Prepare extra parameters for steering
    extra_body = {}
    if sae_index is not None:
        extra_body["sae_index"] = sae_index

    response = await client.chat.completions.create(
        model="thejaminator/sae-introspection-lora",
        messages=[
            {
                "role": "user",
                "content": "Can you explain to me what 'X' means? Format your final answer with <explanation>",
            }
        ],
        max_tokens=500,
        temperature=1.0,
        extra_body=extra_body if extra_body else None,
    )

    return TestChatCompletion(
        sae_index=sae_index,
        response=response.choices[0].message.content,  # type: ignore
    )


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
        # base_url="http://localhost:8000/v1"
        base_url="https://94nlcy6stx75yz-8000.proxy.runpod.net/v1",
    )

    print("Testing vLLM server with activation steering using OpenAI async client...")
    print("Running all requests in parallel...")

    # Create all tasks to run in parallel
    tasks = Slist(
        [
            # Baseline (no steering)
            test_chat_completion(client),
            # Tests with different SAE indices
            test_chat_completion(client, sae_index=0),
            # 10027: good feature?
            test_chat_completion(client, sae_index=10027),
            # 10026 film feature
            test_chat_completion(client, sae_index=10026),
            test_chat_completion(client, sae_index=123),
            test_chat_completion(client, sae_index=999),
        ]
    )

    # Run all tasks in parallel
    results: Slist[TestChatCompletion | None] = await tasks.par_map_async(lambda x: x)

    for result in results:
        assert result is not None
        print(f"\n=== {result.sae_index if result.sae_index is not None else 'Baseline'} ===")
        print(f"Output: {result.response}")


if __name__ == "__main__":
    asyncio.run(main())
