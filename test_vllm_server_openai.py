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
        # model="thejaminator/sae-introspection-lora",
        model="thejaminator/gemma-introspection-20250821",
        messages=[
            {
                "role": "user",
                "content": "Can you explain to me what 'X' means? Format your final answer with <explanation>",
            }
        ],
        max_tokens=1000,
        temperature=1.0,  # stress test passing correct vectors.
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
    saes: Slist[int] = Slist(
        [
            # Baseline (no steering)
            # test_chat_completion(client),
            0,
            1,
            2,
            3,
            10027,
            10026,
            123,
            999,
        ]
    )
    # .repeat_until_size_or_raise(
    #     40
    # )  # this is higher than the max batch size MAX_PARALLEL_REQUESTS of the server of 28. Server should run two generates. let's see what happens?
    print(f"Running {len(saes)} requests")

    # Run all tasks in parallel
    results: Slist[TestChatCompletion | None] = await saes.par_map_async(
        lambda x: test_chat_completion(client, sae_index=x)
    )
    success_results = results.flatten_option()

    for result in success_results.sort_by(lambda x: x.sae_index if x.sae_index is not None else -1):
        # par_map_async / gather is unordered , need to reorder.
        assert result is not None
        print(f"\n=== {result.sae_index if result.sae_index is not None else 'Baseline'} ===")
        print(f"Output: {result.response}")


if __name__ == "__main__":
    asyncio.run(main())
