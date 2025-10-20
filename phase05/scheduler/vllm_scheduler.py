import asyncio
import aiohttp
import os
from typing import Optional

import torch


class VllmScheduler:
    """
    Scheduler that interfaces with vLLM (Virtual LLM) server for text generation.

    This scheduler sends inference requests to a vLLM server endpoint and handles
    the responses asynchronously using aiohttp. It manages connection sessions and
    provides a simple interface for generating text completions.

    Usage:
        # Option 1: Using context manager (recommended)
        async with VllmScheduler() as scheduler:
            result = await scheduler.add_request("Hello, world!")
            print(result)

        # Option 2: Manual initialization and cleanup
        scheduler = VllmScheduler()
        await scheduler.initialize()
        try:
            result = await scheduler.add_request("Hello, world!")
            print(result)
        finally:
            await scheduler.shutdown()
    """

    def __init__(
        self,
        model: str = "gpt2",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        endpoint: str = "http://localhost:8000/v1/completions",
        timeout: int = 10,
    ):
        """
        Initialize the vLLM scheduler with configuration parameters.

        Args:
            model: Name of the LLM model to use (can be overridden by LLM_MODEL env var)
            max_tokens: Maximum number of tokens to generate in responses
            temperature: Sampling temperature for randomness (0.0 = deterministic, higher = more random)
            top_p: Nucleus sampling parameter - cumulative probability threshold
            top_k: Top-k sampling parameter - number of highest probability tokens to consider
            endpoint: URL endpoint of the vLLM server
            timeout: Request timeout in seconds
        """
        # Use environment variable if set, otherwise use provided model parameter
        self.model = os.getenv("LLM_MODEL", model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.session: Optional[aiohttp.ClientSession] = None
        self.endpoint = endpoint
        self.timeout = timeout
        self.memory_usage: Optional[str] = None

    def update_memory_usage(self):
        """
        Update memory usage statistics from CUDA device.

        Calculates peak memory allocated during generation and stores it
        in a human-readable format.
        """
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            self.memory_usage = f"{peak_memory / (1024 ** 3):.2f} GB"
        else:
            self.memory_usage = "N/A (CUDA not available)"

    async def initialize(self):
        """
        Initialize the aiohttp session.

        Returns:
            self: Returns the scheduler instance for method chaining
        """
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self

    async def __aenter__(self):
        """
        Async context manager entry - initializes the aiohttp session.

        This allows the scheduler to be used with 'async with' syntax.

        Returns:
            self: Returns the scheduler instance
        """
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit - closes the aiohttp session.

        Ensures proper cleanup of network resources when exiting the context.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        await self.shutdown()

    async def add_request(self, prompt: str) -> str:
        """
        Send a text generation request to the vLLM server.

        Args:
            prompt: Input text prompt to send to the model

        Returns:
            str: Generated text completion from the model

        Raises:
            RuntimeError: If the session is not initialized
            Exception: If the vLLM server returns a non-200 status code
        """
        # Ensure session is initialized
        if self.session is None:
            raise RuntimeError(
                "Session not initialized. Call 'await scheduler.initialize()' "
                "or use 'async with VllmScheduler() as scheduler:'"
            )

        # Build request payload with model parameters
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        # Send POST request to vLLM server endpoint

        result = await self.process_request(payload=payload)
        return result

    async def process_request(self, payload):
        async with self.session.post(
            self.endpoint,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as response:
            # Check for HTTP errors
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"vLLM HTTP {response.status}: {error_text}")

            # Parse JSON response from vLLM server
            result = await response.json()
            self.update_memory_usage()

            # Extract generated text from the response structure
            # vLLM returns choices array with text in each choice
            generated_text = (
                result["choices"][0]["text"]
                if "choices" in result and len(result["choices"]) > 0
                else ""
            )

            return generated_text

    async def shutdown(self):
        """
        Close the aiohttp session and clean up resources.

        This should be called when done using the scheduler to prevent
        resource leaks.
        """
        if self.session:
            await self.session.close()
            self.session = None


async def main():
    """
    Example usage of the VllmScheduler.
    """
    # Example 1: Using context manager (recommended)
    print("Example 1: Using context manager")
    async with VllmScheduler(temperature=0.8, max_tokens=50) as scheduler:
        try:
            result = await scheduler.add_request("Once upon a time")
            print(f"Generated text: {result}")
            print(f"Memory usage: {scheduler.memory_usage}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 50 + "\n")

    # Example 2: Manual initialization and cleanup
    print("Example 2: Manual initialization")
    scheduler = VllmScheduler(temperature=0.9, max_tokens=75)
    await scheduler.initialize()

    try:
        result = await scheduler.add_request("The quick brown fox")
        print(f"Generated text: {result}")
        print(f"Memory usage: {scheduler.memory_usage}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await scheduler.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
