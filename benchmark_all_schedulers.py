import asyncio
from scheduler.dynamic_scheduler import DynamicScheduler
from scheduler.vllm_scheduler import VllmScheduler
from scheduler.naive_scheduler import NaiveScheduler
import csv
import torch
import json
import os

from typing import List, Optional, Dict, Union, Any

# Dictionary mapping scheduler names to their classes
scheduler_dict: Dict[
    str, Union[type[DynamicScheduler], type[NaiveScheduler], type[VllmScheduler]]
] = {
    "dynamic_scheduler": DynamicScheduler,
    "naive_scheduler": NaiveScheduler,
    "vllm_scheduler": VllmScheduler,
}


def get_scheduler(scheduler_type, n_run, **kwargs):
    """Get scheduler instance based on scheduler type."""
    scheduler_class = scheduler_dict[scheduler_type]

    if scheduler_class == VllmScheduler:
        return VllmScheduler(
            model=kwargs.get("modelname", "gpt2"),
            max_tokens=kwargs.get("max_tokens", 100),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            top_k=kwargs.get("top_k", 50),
            endpoint=kwargs.get("endpoint", "http://localhost:8000/v1/completions"),
            timeout=kwargs.get("timeout", 10),
        )
    elif scheduler_class == DynamicScheduler:
        batch_sizes = kwargs.get("batch_sizes", [2, 4, 8])
        return DynamicScheduler(
            modelname=kwargs.get("modelname", "gpt2"),
            device=kwargs.get("device", "cpu"),
            min_batch_size=kwargs.get("min_batch_size", 2),
            max_batch_size=kwargs.get("max_batch_size", 16),
            batch_size=batch_sizes[n_run],
            timeout=kwargs.get("timeout", 0.5),
            max_time_wait=kwargs.get("max_time_wait", 2.0),
            memory_threshold=kwargs.get("memory_threshold", 0.3),
        )
    elif scheduler_class == NaiveScheduler:
        batch_sizes = kwargs.get("batch_sizes", [2, 4, 8])
        return NaiveScheduler(
            modelname=kwargs.get("modelname", "gpt2"),
            batch_size=batch_sizes[n_run],
            device=kwargs.get("device", "cpu"),
            timeout=kwargs.get("timeout", 0.5),
        )


async def send_request(
    scheduler: Union[DynamicScheduler, NaiveScheduler, VllmScheduler],
    req: Dict[str, Any],
    metrics: List[Dict[str, Any]],
    sem: asyncio.Semaphore,
):
    """Send a request to the scheduler and record metrics."""
    async with sem:
        t0 = asyncio.get_event_loop().time()
        try:
            output = await scheduler.add_request(req["prompt"])
            cleaned_output = " ".join(output.split())

            # Calculate metrics
            total_latency = asyncio.get_event_loop().time() - t0
            ttft = total_latency
            tokens = len(output.split())
            tokens_per_sec = tokens / total_latency if total_latency > 0 else 0
            batch_num = (
                0 if isinstance(scheduler, VllmScheduler) else scheduler.batch_count
            )
            batch_size = (
                0 if isinstance(scheduler, VllmScheduler) else scheduler.batch_size
            )
            current_peak_mem_usage = scheduler.memory_usage

            # Record successful metrics
            metrics.append(
                {
                    "scheduler_type": req["scheduler_type"],
                    "run_num": req["run_num"],
                    "seq": req["seq"],
                    "status": "ok",
                    "batch_num": batch_num,
                    "input_tokens": req["word_count"],
                    "output_tokens": tokens,
                    "total_tokens": req["word_count"] + tokens,
                    "latency": total_latency,
                    "ttft": ttft,
                    "tokens_per_sec": tokens_per_sec,
                    "prompt": req["prompt"],
                    "output_prompt": cleaned_output,
                    "current_batch_size": batch_size,
                    "current_peak_mem_usage": current_peak_mem_usage,
                }
            )
        except Exception as e:
            # Record error
            metrics.append(
                {
                    "scheduler_type": req.get("scheduler_type", ""),
                    "run_num": req["run_num"],
                    "seq": req["seq"],
                    "status": "error",
                    "prompt": req.get("prompt", ""),
                    "error": str(e),
                }
            )


def print_metrics(metrics: List[Dict[str, Any]]) -> None:
    """Display performance metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)

    successful = [m for m in metrics if m["status"] == "ok"]
    errors = [m for m in metrics if m["status"] == "error"]

    print(f"\nTotal requests: {len(metrics)}")
    print(f"Successful: {len(successful)}")
    print(f"Errors: {len(errors)}")

    if successful:
        avg_latency = sum(m["latency"] for m in successful) / len(successful)
        avg_tokens_per_sec = sum(m["tokens_per_sec"] for m in successful) / len(
            successful
        )
        total_batches = max(m["batch_num"] for m in successful)

        print(f"\nAverage latency: {avg_latency:.3f}s")
        print(f"Average tokens/sec: {avg_tokens_per_sec:.2f}")
        print(f"Total batches: {total_batches}")

        # Show batch size evolution
        batch_sizes = [m["current_batch_size"] for m in successful]
        print(f"Batch size range: {min(batch_sizes)} - {max(batch_sizes)}")
        print(f"Final batch size: {batch_sizes[-1]}")

    print("\nDetailed metrics:")
    for m in sorted(metrics, key=lambda x: x["seq"]):
        if m["status"] == "ok":
            print(
                f"  Seq {m['seq']}: {m['output_tokens']} tokens, "
                f"{m['latency']:.3f}s, {m['tokens_per_sec']:.2f} tok/s, "
                f"batch #{m['batch_num']}, batch_size={m['current_batch_size']}"
            )
        else:
            print(f"  Seq {m['seq']}: ERROR - {m['error']}")

    print("=" * 60)


def write_to_csv(metrics: List[Dict[str, Any]], filepath: str) -> None:
    """
    Write metrics to a CSV file with proper column ordering.

    Args:
        metrics: List of metric dictionaries
        filepath: Path to output CSV file
    """
    if not metrics:
        print("No metrics to write!")
        return

    # Define logical column order
    fieldnames = [
        "scheduler_type",
        "run_num",
        "seq",
        "status",
        "batch_num",
        "current_batch_size",
        "current_peak_mem_usage",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency",
        "ttft",
        "tokens_per_sec",
        "prompt",
        "output_prompt",
        "error",
    ]

    # Sort by run number, then sequence number
    sorted_metrics = sorted(
        metrics, key=lambda x: (x.get("run_num", 0), x.get("seq", 0))
    )

    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")

        # Write header
        writer.writeheader()

        # Write rows
        for metric in sorted_metrics:
            writer.writerow(metric)

    print(f"Successfully wrote {len(metrics)} records to {filepath}")


async def main() -> None:
    """Main function to run the benchmark."""

    # Load prompts from JSON file
    prompts_file = os.path.join(os.path.dirname(__file__), "prompts.json")
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    # Configuration
    config_dict = {
        "n_runs": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_concurrent": 10,
        "batch_sizes": [2, 4, 8],
        "min_batch_size": 2,
        "max_batch_size": 32,
        "timeout": 0.5,
        "max_time_wait": 2.0,
        "memory_threshold": 0.9,
        "n_requests": [25, 50, 100],
        "scheduler_types": ["naive_scheduler", "dynamic_scheduler", "vllm_scheduler"],
    }
    all_metrics = []

    for scheduler_type in config_dict["scheduler_types"]:
        for r in range(config_dict["n_runs"]):
            print(f"\n{'='*60}")
            print(
                f"STARTING RUN #{r + 1}/{config_dict['n_runs']} - Scheduler: {scheduler_type}"
            )
            print(f"{'='*60}\n")

            # Initialize scheduler for this run
            scheduler = get_scheduler(scheduler_type, r, **config_dict)

            # CRITICAL FIX: Initialize VllmScheduler session
            if isinstance(scheduler, VllmScheduler):
                await scheduler.initialize()

            # Metrics list for this run
            run_metrics = []

            # Semaphore to limit concurrent requests
            sem = asyncio.Semaphore(config_dict["max_concurrent"])

            # Create request objects (use n_requests to limit prompts)
            n_requests = config_dict["n_requests"][r]
            requests = []
            for i, prompt in enumerate(prompts[:n_requests]):
                request = {
                    "run_num": r + 1,
                    "seq": i + 1,
                    "prompt": prompt,
                    "word_count": len(prompt.split()),
                    "scheduler_type": scheduler_type,
                }
                requests.append(request)

            print(f"Configuration:")
            print(f"  Scheduler: {scheduler_type}")
            print(f"  Prompts: {len(requests)}")
            if not isinstance(scheduler, VllmScheduler):
                print(f"  Initial batch size: {scheduler.batch_size}")
                print(f"  Device: {scheduler.device}")
                print(f"  Timeout: {scheduler.timeout}s")
                if hasattr(scheduler, "max_time_wait"):
                    print(f"  Max wait time: {scheduler.max_time_wait}s")
                if hasattr(scheduler, "memory_threshold"):
                    print(f"  Memory threshold: {scheduler.memory_threshold}")
            else:
                print(f"  Endpoint: {scheduler.endpoint}")
                print(f"  Model: {scheduler.model}")
                print(f"  Max tokens: {scheduler.max_tokens}")
                print(f"  Temperature: {scheduler.temperature}")
            print(f"  Max concurrent: {config_dict['max_concurrent']}")
            print("-" * 60)

            try:
                # Send all requests concurrently
                tasks = [
                    send_request(scheduler, req, run_metrics, sem) for req in requests
                ]
                await asyncio.gather(*tasks)
            finally:
                # CRITICAL: Always shutdown scheduler to prevent resource leaks
                if hasattr(scheduler, "shutdown"):
                    await scheduler.shutdown()

            # Display metrics for this run
            print(f"\nRun #{r + 1} - {scheduler_type} Results:")
            print_metrics(run_metrics)

            # Add to cumulative metrics
            all_metrics.extend(run_metrics)

    # Write all runs to CSV
    print(f"\n{'='*60}")
    print("WRITING ALL RUNS TO CSV")
    print(f"{'='*60}")
    write_to_csv(all_metrics, "llm_inference_benchmark.csv")

    # Summary statistics across all runs
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL RUNS")
    print(f"{'='*60}")
    print(f"Total requests: {len(all_metrics)}")

    successful = [m for m in all_metrics if m["status"] == "ok"]
    if successful:
        avg_latency = sum(m["latency"] for m in successful) / len(successful)
        avg_tokens_per_sec = sum(m["tokens_per_sec"] for m in successful) / len(
            successful
        )

        print(f"Successful: {len(successful)}")
        print(f"Overall average latency: {avg_latency:.3f}s")
        print(f"Overall average tokens/sec: {avg_tokens_per_sec:.2f}")

        # Batch size statistics (only for non-vllm schedulers)
        non_vllm_successful = [m for m in successful if m["current_batch_size"] > 0]
        if non_vllm_successful:
            batch_sizes = [m["current_batch_size"] for m in non_vllm_successful]
            print(
                f"Batch size range across all runs: {min(batch_sizes)} - {max(batch_sizes)}"
            )

    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
