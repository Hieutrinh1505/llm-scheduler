import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import torch


class AsyncDynamicScheduler:
    """
    Asynchronous batch scheduler with dynamic batch sizing.

    Features:
    - Adaptive batch size based on GPU memory usage
    - Timeout-based batch collection
    - Max wait time guarantee for individual requests
    """

    def __init__(
        self,
        modelname: str,
        device: str = "cpu",
        min_batch_size: int = 4,
        max_batch_size: int = 32,
        batch_size: int = 8,
        timeout: float = 0.5,
        max_time_wait: float = 2.0,
        memory_threshold: float = 0.8,
    ):
        """
        Initialize the dynamic scheduler.

        Args:
            modelname: HuggingFace model identifier
            device: Device to run inference on ('cpu' or 'cuda')
            min_batch_size: Minimum batch size (lower bound for adaptation)
            max_batch_size: Maximum batch size (upper bound for adaptation)
            batch_size: Initial batch size
            timeout: Max seconds to wait for batch to fill before processing
            max_time_wait: Max seconds first request can wait (latency SLA)
            memory_threshold: GPU memory usage ratio to trigger batch size reduction
        """
        self.modelname = modelname
        self.device = device
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_time_wait = max_time_wait
        self.memory_threshold = memory_threshold

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        self.model = AutoModelForCausalLM.from_pretrained(self.modelname)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)

        # Initialize queue and state
        self.queue = asyncio.Queue()
        self.batch_count = 0
        self.running = True
        self.memory_usage = None

        # Start background batch processing loop
        self.task = asyncio.create_task(self._batch_loop())

    async def add_request(self, prompt: str) -> str:
        """
        Add a request to the processing queue.

        Args:
            prompt: Input text prompt

        Returns:
            Generated text (awaits until batch is processed)
        """
        # Create a future to wait for the result
        fut = asyncio.get_event_loop().create_future()

        # Add (prompt, future) pair to queue
        await self.queue.put((prompt, fut))

        # Wait for result (blocks until batch processing completes)
        return await fut

    async def _batch_loop(self):
        """
        Background loop that continuously processes batches.

        Exits when:
        1. Batch reaches target size
        2. Timeout expires (waited long enough for batch to fill)
        3. First request exceeds max_time_wait (latency SLA)
        """
        while self.running:
            batch = []
            first_request_time = None

            # Get current queue size for adaptive batching
            target_batch_size = min(self.batch_size, self.max_batch_size)

            # Calculate target batch size based on current conditions
            target_batch_size = min(
                self.batch_size,  # Current adaptive batch size
                self.max_batch_size,  # Hard upper limit
            )

            # Record when batch collection starts
            batch_start_time = asyncio.get_event_loop().time()

            try:
                # Collect requests until target size or timeout
                while len(batch) < target_batch_size:
                    # Calculate remaining time for batch collection
                    timeout_remaining = self.timeout - (
                        asyncio.get_event_loop().time() - batch_start_time
                    )

                    # Exit if timeout expired and we have items to process
                    if timeout_remaining <= 0 and batch:
                        break

                    try:
                        # Wait for next item with remaining timeout
                        item = await asyncio.wait_for(
                            self.queue.get(), timeout=max(0.01, timeout_remaining)
                        )
                        batch.append(item)

                        # Track when FIRST request arrived (for latency SLA)
                        if first_request_time is None:
                            first_request_time = asyncio.get_event_loop().time()

                        # Check if first request has waited too long
                        if first_request_time is not None:
                            wait_time = (
                                asyncio.get_event_loop().time() - first_request_time
                            )
                            if wait_time >= self.max_time_wait:
                                # Flush batch immediately to meet latency SLA
                                break

                    except asyncio.TimeoutError:
                        # Timeout reached while waiting for next item
                        if batch:
                            # Process what we have
                            break
                        # No items yet, continue waiting
                        continue

                # Skip if no items collected
                if not batch:
                    continue

                # Separate prompts and futures
                prompts = [prompt for prompt, _ in batch]
                futures = [fut for _, fut in batch]

                # Run inference on the batch
                decoded = self.run_batch(prompts)

                # Set results for all futures
                self._set_results(futures, decoded)

                self.batch_count += 1

            except Exception as e:
                # Handle errors by setting exception on all futures
                print(f"Error in batch loop: {e}")
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)

    def _set_results(self, futures: List, decoded: List[str]) -> None:
        """
        Set results for all futures in the batch.

        Args:
            futures: List of asyncio futures
            decoded: List of generated text outputs
        """
        for fut, out in zip(futures, decoded):
            if not fut.done():
                fut.set_result(out)

    def run_batch(self, prompts: List[str]) -> List[str]:
        """
        Process a batch of prompts with adaptive batch sizing.

        Monitors GPU memory usage and adjusts batch_size accordingly:
        - High memory usage (>threshold): Decrease batch size
        - Low memory usage (<threshold): Increase batch size

        Args:
            prompts: List of input prompts

        Returns:
            List of generated text (one per prompt)
        """
        # Reset peak memory stats for this batch (CUDA only)
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Tokenize all prompts with padding
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.device
        )

        # Store input length to extract only new tokens later
        input_length = inputs["input_ids"].shape[1]

        # Generate text
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=20,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )

        # Adaptive batch sizing based on memory usage (CUDA only)
        if self.device == "cuda":
            # Get peak memory used during generation
            peak_memory = torch.cuda.max_memory_allocated()
            _, total_memory = torch.cuda.mem_get_info()

            self.memory_usage = f"{peak_memory / (1024 ** 3):.2f} GB"

            # Calculate memory usage ratio
            memory_usage_ratio = peak_memory / total_memory

            # Adjust batch size based on memory pressure
            if memory_usage_ratio >= self.memory_threshold:
                # High memory usage: reduce batch size conservatively
                self.batch_size = max(self.min_batch_size, self.batch_size - 1)
            else:
                # Low memory usage: increase batch size more aggressively
                self.batch_size = min(self.batch_size + 2, self.max_batch_size)

        # Extract only newly generated tokens (exclude input prompt)
        new_tokens = outputs[:, input_length:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        return decoded

    async def shutdown(self):
        """
        Gracefully shutdown the scheduler.
        Processes any remaining items in queue before stopping.
        """
        # Signal the loop to stop
        self.running = False

        # Collect remaining items from queue
        remaining_batch = []
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                remaining_batch.append(item)
            except asyncio.QueueEmpty:
                break

        # Process remaining items if any
        if remaining_batch:
            try:
                prompts = [prompt for prompt, _ in remaining_batch]
                futures = [fut for _, fut in remaining_batch]
                decoded = self.run_batch(prompts)

                # Set results for remaining futures
                self._set_results(futures, decoded)

            except Exception as e:
                # Set exception on all remaining futures
                print(f"Error during shutdown: {e}")
                for _, fut in remaining_batch:
                    if not fut.done():
                        fut.set_exception(e)

        # Cancel and await the background task
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
