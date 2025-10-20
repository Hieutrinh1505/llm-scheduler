# Phase 05: Comprehensive Scheduler Benchmark Suite

A complete benchmarking framework for comparing different LLM scheduler implementations across various workload patterns.

## Overview

This phase provides a comprehensive testing and comparison framework for three scheduler implementations:

1. **Naive Scheduler** - Synchronous, fixed batch size (from Phase 03)
2. **Dynamic Scheduler** - Asynchronous, adaptive batch sizing (from Phase 04)
3. **vLLM Scheduler** - Server-based, continuous batching (production-grade)

## Features

- **Multi-Load Testing**: Test with 25, 50, 100, 200, or 500+ concurrent requests
- **Comprehensive Metrics**: Latency (P50/P95/P99), throughput (req/s, tokens/s), batch efficiency
- **Automated Analysis**: Statistical summaries, comparison tables, visualizations
- **Flexible Configuration**: Customize request sizes, run counts, schedulers to test

## Directory Structure

```
phase05/
├── scheduler/
│   ├── naive_scheduler.py         # Simple synchronous scheduler
│   ├── dynamic_scheduler.py       # Adaptive async scheduler
│   ├── vllm_scheduler.py          # vLLM client wrapper
│   └── start_server.sh            # Helper to start vLLM server
├── benchmark_all_schedulers.py    # Main benchmark script
├── plot_benchmark_results.py      # Visualization and analysis
├── pyproject.toml                 # Dependencies
└── README.md                      # This file
```

## Quick Start

### 1. Install Dependencies

```bash
cd phase05
uv sync
# or
pip install -r pyproject.toml
```

### 2. Run Benchmarks

**Basic usage (Naive + Dynamic schedulers, 25/50/100 requests):**

```bash
python benchmark_all_schedulers.py
```

**Custom configuration:**

```bash
# Test with 25, 50, 100, and 200 requests, 5 runs each
python benchmark_all_schedulers.py --sizes 25 50 100 200 --runs 5

# Test only dynamic scheduler with 100 and 500 requests
python benchmark_all_schedulers.py --schedulers dynamic --sizes 100 500

# Test all three schedulers (requires vLLM server running)
python benchmark_all_schedulers.py --schedulers naive dynamic vllm --sizes 50 100
```

**Command-line options:**

```bash
--sizes           Request sizes to test (default: 25 50 100)
--runs            Number of runs per configuration (default: 3)
--device          Device to use: cuda or cpu (default: cuda)
--max-concurrent  Max concurrent requests (default: 10)
--schedulers      Schedulers to test: naive, dynamic, vllm (default: naive dynamic)
--output-dir      Output directory for results (default: .)
```

### 3. Visualize Results

After running benchmarks, generate plots and analysis:

```bash
python plot_benchmark_results.py
```

This creates:
- `throughput_comparison.png` - Throughput (req/s and tokens/s) vs request size
- `latency_comparison.png` - Latency percentiles (P50, P95, P99, Avg)
- `latency_distribution.png` - Box plots of latency distributions
- `scheduler_comparison_bars.png` - Bar charts comparing metrics
- `benchmark_report.txt` - Detailed text report

**Custom visualization:**

```bash
python plot_benchmark_results.py \
  --summaries benchmark_summaries.csv \
  --metrics benchmark_detailed_metrics.csv \
  --output-dir ./plots
```

## Testing with vLLM

To benchmark against vLLM's production-grade scheduler:

### 1. Configure Environment

Create or edit `.env` file in the `phase05` directory:

```bash
# Copy example file
cp .env.example .env

# Edit with your settings
# .env file contents:
LLM_MODEL=gpt2                                      # Model to use
VLLM_ENDPOINT=http://localhost:8000/v1/completions # Server endpoint
VLLM_PORT=8000                                      # Server port
VLLM_HOST=0.0.0.0                                   # Server host
GPU_MEMORY_UTIL=0.9                                 # GPU memory utilization
```

**Supported models:**
- `gpt2` (small, fast, good for testing)
- `gpt2-medium` (better quality)
- `gpt2-large` (best quality, slower)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (very fast)
- Any HuggingFace model compatible with vLLM

### 2. Start vLLM Server

The `start_server.sh` script automatically reads from `.env`:

```bash
# Start server (in separate terminal)
cd scheduler
./start_server.sh

# The script will:
# - Load configuration from .env
# - Display settings (model, port, etc.)
# - Check for CUDA availability
# - Start vLLM server
```

**Manual start (alternative):**
```bash
vllm serve gpt2 --port 8000 --host 0.0.0.0
```

### 3. Run Benchmark with vLLM

```bash
python benchmark_all_schedulers.py --schedulers naive dynamic vllm --sizes 50 100 200
```

## Understanding the Output

### Console Output

During benchmarking, you'll see real-time progress:

```
======================================================================
TESTING WITH 50 REQUESTS
======================================================================

--- Run 1/3 ---

Benchmarking Naive Scheduler: 50 requests, run 1

NAIVE SCHEDULER - Run 1
============================================================
Request Size: 50
Total Requests: 50
Successful: 50
Failed: 0
Wall Time: 12.34s

Throughput:
  Requests/sec: 4.05
  Tokens/sec: 45.23

Latency:
  P50: 0.234s
  P95: 0.456s
  P99: 0.678s
  Avg: 0.289s
  Min: 0.123s
  Max: 0.789s
```

### Final Comparison Table

At the end, a summary table compares all schedulers:

```
================================================================================
SCHEDULER COMPARISON SUMMARY
================================================================================

50 Requests:
--------------------------------------------------------------------------------
Scheduler       Throughput (req/s)   Throughput (tok/s)   P95 Latency (s)
--------------------------------------------------------------------------------
naive           4.05                 45.23                0.456
dynamic         8.12                 92.15                0.298
vllm            15.34                178.92               0.156
```

### Output Files

**`benchmark_summaries.csv`** - Summary statistics per run:
- scheduler, request_size, run_num
- throughput (req/s, tokens/s)
- latency percentiles (P50, P95, P99, avg, min, max)
- success/failure counts

**`benchmark_detailed_metrics.csv`** - Per-request detailed metrics:
- Individual request latency
- Input/output token counts
- Tokens per second
- Prompt and generated text
- Error messages (if any)

## Interpreting Results

### Key Metrics

**Throughput (requests/sec):**
- Higher is better
- Measures how many requests the scheduler can process per second
- Important for system capacity planning

**Throughput (tokens/sec):**
- Higher is better
- Measures actual token generation speed
- Better indicator of GPU utilization than req/s

**P95 Latency:**
- Lower is better
- 95% of requests complete faster than this time
- Standard SLA metric for production systems

**P99 Latency:**
- Lower is better
- 99% of requests complete faster than this time
- Indicates worst-case performance for most users

### Expected Performance Characteristics

**Naive Scheduler:**
- ✓ Simplest implementation
- ✓ Predictable batch behavior
- ✗ Fixed batch size (inefficient with variable load)
- ✗ Synchronous (lower concurrency)
- **Best for:** Learning, prototyping, batch processing

**Dynamic Scheduler:**
- ✓ Adaptive batching (better GPU utilization)
- ✓ Latency SLA enforcement
- ✓ Memory-aware scaling
- ✗ More complex implementation
- **Best for:** Variable workloads, resource-constrained environments

**vLLM Scheduler:**
- ✓ Production-grade continuous batching
- ✓ Highest throughput
- ✓ Lowest latency
- ✓ PagedAttention memory optimization
- ✗ Requires separate server process
- **Best for:** Production deployments, high-throughput scenarios

## Example Benchmark Workflow

```bash
# 1. Run comprehensive benchmark (3 runs each, multiple sizes)
python benchmark_all_schedulers.py \
  --schedulers naive dynamic \
  --sizes 25 50 100 200 \
  --runs 3 \
  --device cuda

# 2. Generate visualizations
python plot_benchmark_results.py

# 3. Review results
cat benchmark_report.txt
open throughput_comparison.png
open latency_comparison.png
```

## Configuration Tips

### For Latency-Sensitive Workloads

```bash
# Use smaller batch sizes, more concurrent requests
python benchmark_all_schedulers.py \
  --schedulers dynamic \
  --sizes 50 100 \
  --max-concurrent 20
```

### For Throughput-Focused Testing

```bash
# Use larger request sizes, fewer concurrent
python benchmark_all_schedulers.py \
  --schedulers dynamic vllm \
  --sizes 200 500 1000 \
  --max-concurrent 5
```

### For Memory-Constrained GPUs

```bash
# Use CPU or test with smaller loads
python benchmark_all_schedulers.py \
  --device cpu \
  --sizes 10 25 50 \
  --max-concurrent 5
```

## Customizing Tests

### Add Custom Prompts

Edit `benchmark_all_schedulers.py`:

```python
TEST_PROMPTS = [
    "Your custom prompt 1",
    "Your custom prompt 2",
    # ... add more
]
```

### Adjust Scheduler Parameters

Modify scheduler initialization in the benchmark methods:

```python
scheduler = AsyncDynamicScheduler(
    modelname="gpt2",
    batch_size=16,        # Adjust initial batch size
    min_batch_size=8,     # Adjust min
    max_batch_size=64,    # Adjust max
    timeout=1.0,          # Longer timeout
    max_time_wait=3.0,    # Longer SLA
)
```

## Troubleshooting

**CUDA out of memory:**
```bash
# Use smaller batch sizes or fewer concurrent requests
python benchmark_all_schedulers.py --device cpu --sizes 25 50
```

**vLLM connection refused:**
```bash
# Ensure vLLM server is running
cd scheduler
./start_server.sh

# Check server is accessible
curl http://localhost:8000/v1/models
```

**Benchmark runs too slowly:**
```bash
# Reduce number of runs and request sizes
python benchmark_all_schedulers.py --sizes 25 50 --runs 2
```

**Import errors:**
```bash
# Ensure all dependencies are installed
uv sync
# or
pip install torch transformers vllm matplotlib aiohttp python-dotenv
```

## Advanced Usage

### Compare Different Models

```python
# Modify scheduler creation to use different models
scheduler = AsyncDynamicScheduler(
    modelname="gpt2-medium",  # or gpt2-large, EleutherAI/gpt-neo-125M, etc.
    # ... other params
)
```

### Export to Different Format

The CSV files can be easily processed with pandas:

```python
import pandas as pd

# Load results
df = pd.read_csv('benchmark_summaries.csv')

# Group and analyze
grouped = df.groupby(['scheduler', 'request_size'])
print(grouped['throughput_req_per_sec'].mean())

# Export to Excel
df.to_excel('results.xlsx', index=False)
```

### Integrate with CI/CD

```bash
# Run benchmarks and check for regressions
python benchmark_all_schedulers.py --runs 2 --sizes 50

# Parse results programmatically
python -c "
import csv
with open('benchmark_summaries.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['scheduler'] == 'dynamic':
            assert float(row['throughput_req_per_sec']) > 5.0
            print('✓ Dynamic scheduler meets performance threshold')
"
```

## Performance Expectations

Based on testing with GPT-2 on an NVIDIA A100:

| Scheduler | 50 Req (req/s) | 100 Req (req/s) | 200 Req (req/s) | P95 Latency (50 req) |
|-----------|----------------|-----------------|-----------------|---------------------|
| Naive     | ~4-6           | ~5-7            | ~6-8            | ~0.4-0.6s           |
| Dynamic   | ~8-12          | ~10-14          | ~12-16          | ~0.2-0.4s           |
| vLLM      | ~15-25         | ~20-30          | ~25-35          | ~0.1-0.2s           |

*Note: Actual numbers vary based on hardware, model size, and prompt complexity*

## Next Steps

After analyzing benchmark results:

1. **Optimize parameters** - Tune batch sizes, timeouts based on your workload
2. **Profile bottlenecks** - Use detailed metrics to identify slow requests
3. **Scale testing** - Test with larger request sizes (500, 1000+)
4. **Multi-GPU** - Extend to test tensor parallelism configurations
5. **Production deployment** - Use insights to configure production scheduler

## References

- [Phase 03 README](../phase03/README.md) - Naive Scheduler details
- [Phase 04 README](../phase04/README.md) - Dynamic Scheduler details
- [vLLM Documentation](https://docs.vllm.ai/) - vLLM scheduler internals

## Contributing

To add new schedulers to the benchmark:

1. Implement scheduler with compatible interface:
   - `async def add_request(prompt: str) -> str`
   - `async def shutdown()`

2. Add benchmark method in `SchedulerBenchmark` class

3. Update `run_benchmarks()` to include new scheduler

4. Add visualization support in `plot_benchmark_results.py`
