import asyncio
from async_dynamic_scheduler import AsyncDynamicScheduler
from typing import Dict, Any, List
import torch
import csv


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


async def send_request(
    req: Dict[str, Any],
    scheduler: AsyncDynamicScheduler,
    metrics: List[Dict[str, Any]],
    sem: asyncio.Semaphore,
) -> None:
    """
    Send a single request to the scheduler and track metrics.

    Args:
        req: Request dictionary with 'run_num', 'seq', 'prompt', 'word_count'
        scheduler: The AsyncDynamicScheduler instance
        metrics: Shared list to collect performance metrics
        sem: Semaphore to limit concurrent requests
    """
    async with sem:
        # Record start time
        t0 = asyncio.get_event_loop().time()

        try:
            # Send request and wait for result
            output = await scheduler.add_request(req["prompt"])

            # Clean output: remove newlines and extra whitespace
            cleaned_output = " ".join(output.split())

            # Calculate metrics
            total_latency = asyncio.get_event_loop().time() - t0
            ttft = total_latency  # Time to first token (for naive impl, same as total)
            tokens = len(output.split())  # Rough token count
            tokens_per_sec = tokens / total_latency if total_latency > 0 else 0
            batch_num = scheduler.batch_count
            batch_size = scheduler.batch_size
            current_peak_mem_usage = scheduler.memory_usage

            # Record successful metrics
            metrics.append(
                {
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
                    "run_num": req["run_num"],
                    "seq": req["seq"],
                    "status": "error",
                    "prompt": req.get("prompt", ""),
                    "error": str(e),
                }
            )


async def main() -> None:
    """Main function to run the benchmark."""

    # 30 diverse prompts
    prompts = [
        "The future of AI is",
        "Once upon a time",
        "In a galaxy far away",
        "Python is great because",
        "Data science is",
        "Machine learning helps us",
        "Deep neural networks can",
        "The weather today is",
        "Climate change affects",
        "Renewable energy sources include",
        "The history of computers began",
        "Quantum computing will",
        "Space exploration has",
        "The human brain is",
        "Artificial intelligence can be used to",
        "The internet changed society by",
        "Virtual reality technology allows",
        "Cybersecurity is important because",
        "Cloud computing enables",
        "Natural language processing helps",
        "Blockchain technology can",
        "Self-driving cars will",
        "Biotechnology advances have",
        "The scientific method involves",
        "Education in the future will",
        "Social media platforms have",
        "Environmental conservation requires",
        "Economic systems are based on",
        "Democracy depends on",
        "Healthcare innovations include",
    ]

    # Configuration
    n_runs = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_concurrent = 10
    batch_sizes = [4,6,8]
    # Store metrics for all runs
    all_metrics = []

    for r in range(n_runs):
        print(f"\n{'='*60}")
        print(f"STARTING RUN #{r + 1}/{n_runs}")
        print(f"{'='*60}\n")

        # Initialize scheduler for this run
        scheduler = AsyncDynamicScheduler(
            modelname="gpt2",
            device=device,
            min_batch_size=2,
            max_batch_size=16,
            batch_size=batch_sizes[r],
            timeout=0.5,
            max_time_wait=2.0,
            memory_threshold=0.3,  # ← Change this only
        )

        # Metrics list for this run
        run_metrics = []

        # Semaphore to limit concurrent requests
        sem = asyncio.Semaphore(max_concurrent)

        # Create request objects
        requests = []
        for i, prompt in enumerate(prompts):
            request = {
                "run_num": r + 1,
                "seq": i + 1,
                "prompt": prompt,
                "word_count": len(prompt.split()),
            }
            requests.append(request)

        print(f"Configuration:")
        print(f"  Prompts: {len(requests)}")
        print(f"  Initial batch size: {scheduler.batch_size}")
        print(
            f"  Min/Max batch size: {scheduler.min_batch_size}/{scheduler.max_batch_size}"
        )
        print(f"  Device: {scheduler.device}")
        print(f"  Max concurrent: {max_concurrent}")
        print(f"  Timeout: {scheduler.timeout}s")
        print(f"  Max wait time: {scheduler.max_time_wait}s")
        print(f"  Memory threshold: {scheduler.memory_threshold}")
        print("-" * 60)

        # Send all requests concurrently
        tasks = [send_request(req, scheduler, run_metrics, sem) for req in requests]
        await asyncio.gather(*tasks)

        # Shutdown scheduler
        await scheduler.shutdown()

        # Display metrics for this run
        print(f"\nRun #{r + 1} Results:")
        print_metrics(run_metrics)

        # Add to cumulative metrics
        all_metrics.extend(run_metrics)

    # Write all runs to CSV
    print(f"\n{'='*60}")
    print("WRITING ALL RUNS TO CSV")
    print(f"{'='*60}")
    write_to_csv(all_metrics, "dynamic_benchmark_results.csv")

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

        # Batch size statistics
        batch_sizes = [m["current_batch_size"] for m in successful]
        print(
            f"Batch size range across all runs: {min(batch_sizes)} - {max(batch_sizes)}"
        )

    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
