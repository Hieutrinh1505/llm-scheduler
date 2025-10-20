"""
Test script for vLLM server connectivity and basic functionality.

This script verifies that:
1. vLLM server is running and accessible
2. OpenAI-compatible API is working correctly
3. Metrics endpoint is available

Usage:
    python test_vllm.py

Prerequisites:
    - vLLM server must be running (see scheduler/start_server.sh)
    - Server should be accessible at http://localhost:8000
"""

import asyncio
from openai import OpenAI
import aiohttp


def test_completion():
    """
    Test synchronous text completion using OpenAI-compatible API.

    This tests the /v1/completions endpoint which is used by VllmScheduler.
    """
    print("Testing vLLM Completion API...")
    print("=" * 60)

    # Create OpenAI client pointing to vLLM server
    # API key is not needed for local vLLM server
    client = OpenAI(api_key="no-need", base_url="http://localhost:8000/v1")

    # Send a completion request
    response = client.completions.create(
        model="gpt2",
        prompt="Tell me a joke",
        max_tokens=100,
        temperature=0.7
    )

    # Display the generated text
    print("✓ Connection successful!")
    print(f"Generated text: {response.choices[0].text}")
    print("=" * 60 + "\n")


async def get_vllm_metrics():
    """
    Fetch vLLM server metrics asynchronously.

    Metrics include:
    - Request counts and latencies
    - Token throughput
    - Memory usage
    - Active requests
    """
    print("Fetching vLLM Server Metrics...")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/metrics") as response:
            metrics = await response.text()
            print("✓ Metrics endpoint accessible!")
            print("\nServer Metrics:")
            print(metrics)
            print("=" * 60)


async def main():
    """Main test function."""
    print("\n" + "=" * 60)
    print("vLLM Server Test Suite")
    print("=" * 60 + "\n")

    try:
        # Test 1: Completion API
        test_completion()

        # Test 2: Metrics endpoint
        await get_vllm_metrics()

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure vLLM server is running: cd scheduler && ./start_server.sh")
        print("2. Check server is accessible: curl http://localhost:8000/v1/models")
        print("3. Verify port 8000 is not blocked")


# Run the test suite
if __name__ == "__main__":
    asyncio.run(main())
