# LLM Scheduler Benchmark Suite

A comprehensive benchmarking framework comparing three LLM inference schedulers: Naive, Dynamic, and vLLM.

Compare batch processing strategies, measure performance metrics (TTFT, throughput, latency), and generate detailed visualizations.

---

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Hieutrinh1505/llm-scheduler.git
cd llm-scheduler

# Install dependencies with UV (recommended)
uv sync

# Or use pip
pip install -e .
```

### 2. Run Benchmark

```bash
# Basic benchmark (Naive + Dynamic schedulers)
python benchmark_all_schedulers.py

# With all schedulers (requires vLLM server - see below)
python benchmark_all_schedulers.py  # runs all 3 schedulers
```

### 3. Generate Plots

```bash
# Create 7 visualization charts
python benchmark_plot.py
```

All plots saved to `plots/` directory.

---

## 📁 Project Structure

```
llm-scheduler/
├── scheduler/                   # Scheduler implementations
│   ├── naive_scheduler.py          # Fixed batch size, sequential processing
│   ├── dynamic_scheduler.py        # Adaptive batching with SLA enforcement
│   ├── vllm_scheduler.py           # vLLM client (continuous batching)
│   └── start_server.sh             # vLLM server startup script
├── tests/
│   └── test_vllm.py               # vLLM connectivity test
├── plots/                         # Generated visualizations
├── benchmark_all_schedulers.py    # Main benchmarking script
├── benchmark_plot.py              # Visualization generator
├── prompts.json                   # Test prompts dataset
├── pyproject.toml                 # Dependencies (uv/pip)
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

---

## 🧠 Scheduler Overview

### 1. Naive Scheduler (`naive_scheduler.py`)

**Core Concept:** Simple batch processing with fixed batch size.

**How it works:**
- Collects requests into batches up to `batch_size`
- Waits for batch to fill OR timeout expires
- Processes entire batch synchronously
- Sequential execution (one batch at a time)

**Key Features:**
- ✓ Simple, predictable behavior
- ✓ Good for learning batch processing concepts
- ✗ Fixed batch size (inefficient under variable load)
- ✗ No memory adaptation

**When to use:** Learning, debugging, batch processing jobs

**Configuration:**
```python
NaiveScheduler(
    modelname="gpt2",
    batch_size=4,      # Fixed batch size
    device="cuda",
    timeout=0.5        # Max wait time for batch
)
```

---

### 2. Dynamic Scheduler (`dynamic_scheduler.py`)

**Core Concept:** Adaptive batching with memory-aware scaling and latency SLA.

**How it works:**
- Dynamically adjusts batch size based on GPU memory usage
- Enforces `max_time_wait` SLA (no request waits too long)
- Monitors memory pressure and scales batch size up/down
- Asynchronous processing with concurrent request handling

**Key Features:**
- ✓ Adaptive batch sizing (better GPU utilization)
- ✓ Memory-aware: reduces batch size when memory high
- ✓ Latency SLA enforcement (`max_time_wait`)
- ✓ Handles variable workload efficiently
- ✗ More complex than naive scheduler

**When to use:** Variable workloads, resource-constrained GPUs, latency-sensitive apps

**Configuration:**
```python
DynamicScheduler(
    modelname="gpt2",
    batch_size=8,           # Initial batch size
    min_batch_size=4,       # Lower bound
    max_batch_size=32,      # Upper bound
    timeout=0.5,            # Batch collection timeout
    max_time_wait=2.0,      # Latency SLA (max wait per request)
    memory_threshold=0.8    # GPU memory trigger for scaling
)
```

**Adaptive Logic:**
- High memory (>80%): Decrease batch size by 1
- Low memory (<80%): Increase batch size by 2
- Always respects `min_batch_size` and `max_batch_size`

---

### 3. vLLM Scheduler (`vllm_scheduler.py`)

**Core Concept:** Production-grade continuous batching via vLLM server.

**How it works:**
- Client-server architecture (scheduler sends HTTP requests to vLLM)
- vLLM uses **PagedAttention** for efficient KV cache management
- **Continuous batching:** New requests join in-progress batches
- No waiting for batch to fill - immediate processing

**Key Features:**
- ✓ Highest throughput (continuous batching)
- ✓ Lowest latency (no batch wait time)
- ✓ Production-ready optimizations (PagedAttention, kernel fusion)
- ✓ Supports large models efficiently
- ✗ Requires separate server process
- ✗ More complex setup

**When to use:** Production deployments, high-throughput scenarios, serving at scale

**Setup:**
```bash
# 1. Start vLLM server (in separate terminal)
cd scheduler
./start_server.sh

# 2. Test connection
cd ..
python tests/test_vllm.py

# 3. Run benchmark with vLLM
python benchmark_all_schedulers.py
```

**Configuration:**
```python
VllmScheduler(
    model="gpt2",
    endpoint="http://localhost:8000/v1/completions",
    max_tokens=100,
    temperature=0.7
)
```

---

## 📊 Benchmark Script (`benchmark_all_schedulers.py`)

**Purpose:** Compare scheduler performance across different request loads.

**What it does:**
1. Loads test prompts from `prompts.json`
2. Runs each scheduler with varying request counts (25, 50, 99)
3. Measures per-request metrics (latency, TTFT, throughput)
4. Exports results to `llm_inference_benchmark.csv`

**Key Metrics Collected:**
- **Latency:** Total time from request to completion
- **TTFT (Time to First Token):** Time until first token generated
- **Throughput:** Tokens per second
- **Batch metrics:** Batch number, batch size
- **Memory usage:** Peak GPU memory

**Configuration:**
```python
config_dict = {
    "n_runs": 3,                    # 3 runs with different request counts
    "device": "cuda",               # Use GPU
    "max_concurrent": 10,           # Max concurrent requests
    "batch_sizes": [2, 4, 8],       # Batch sizes for each run
    "n_requests": [25, 50, 99],     # Request counts to test
    "scheduler_types": [            # Schedulers to benchmark
        "naive_scheduler",
        "dynamic_scheduler",
        "vllm_scheduler"
    ]
}
```

**Output File:** `llm_inference_benchmark.csv`
- Columns: scheduler_type, run_num, seq, latency, ttft, tokens_per_sec, batch_size, etc.

---

## 📈 Visualization (`benchmark_plot.py`)

**Purpose:** Generate visual comparisons of scheduler performance.

**Generated Plots (7 total):**

### Bar Charts (grouped comparison)
1. **`ttft_bar_chart.png`** - Average TTFT by scheduler and request count
2. **`throughput_bar_chart.png`** - Tokens/sec by scheduler
3. **`total_latency_bar_chart.png`** - Total completion time

### Line Charts (trend analysis)
4. **`ttft_line_chart.png`** - TTFT trends across request loads
5. **`throughput_line_chart.png`** - Throughput scaling

### Distribution Analysis
6. **`latency_boxplot.png`** - Latency distribution (P50, P95, P99)
7. **`latency_percentiles.png`** - P50/P95/P99 comparison across schedulers

**Usage:**
```python
from benchmark_plot import BenchmarkPlot

# Initialize with CSV file
plotter = BenchmarkPlot("llm_inference_benchmark.csv")

# Generate individual plots
plotter.plot_ttft()              # Bar chart
plotter.plot_ttft_line()         # Line chart
plotter.plot_latency_boxplot()   # Box plot
plotter.plot_latency_percentiles() # Percentile comparison

# Or run all plots
python benchmark_plot.py
```

---

## 🔑 Key Takeaways

### Performance Characteristics

| Metric | Naive | Dynamic | vLLM |
|--------|-------|---------|------|
| **TTFT (avg)** | 0.7s @ 25 req<br>0.3s @ 50 req<br>**0.18s @ 99 req** | 0.33s @ 25 req<br>0.39s @ 50 req<br>0.55s @ 99 req | **0.23s** (consistent) |
| **Throughput** | Low (sequential) | Medium (adaptive) | **Highest** |
| **Memory Efficiency** | Fixed allocation | ✓ Adaptive | ✓ PagedAttention |
| **Latency SLA** | ✗ No guarantee | ✓ Enforced | ✓ Low variance |
| **Best Use Case** | Learning/Batch | Variable load | **Production** |

### Surprising Finding: Naive Scheduler TTFT at 99 Requests

**Why is Naive fastest at high load?**
- **No batching overhead:** Processes requests immediately one-by-one
- **No wait time:** Each request starts processing right away
- **Dynamic/vLLM wait for batches:** Incur batching delay

**BUT...**
- **Naive has WORST total throughput** (slowest to complete all 99 requests)
- **TTFT ≠ Total Latency:** First token is fast, but total completion is slow
- **Trade-off:** Low TTFT vs. High Throughput

**Conclusion:** Naive's low TTFT at high load is misleading - check total latency/throughput plots!

---

### When to Use Each Scheduler

**Naive Scheduler:**
- 📚 Learning batch processing concepts
- 🧪 Prototyping and debugging
- 📦 Offline batch processing (non-interactive)
- 💡 Simple, predictable behavior needed

**Dynamic Scheduler:**
- 📊 Variable workload patterns
- 💾 Limited GPU memory
- ⏱️ Latency SLA requirements
- 🔄 Need adaptive resource management

**vLLM Scheduler:**
- 🚀 Production serving at scale
- 📈 High throughput requirements
- 🎯 Lowest latency needed
- 💪 Large models (efficient memory with PagedAttention)

---

## 🛠️ Advanced Usage

### Custom Batch Sizes

Edit `benchmark_all_schedulers.py`:
```python
config_dict = {
    "batch_sizes": [4, 8, 16],  # Test different sizes
    "n_requests": [50, 100, 200]
}
```

### Custom Prompts

Replace `prompts.json` with your own dataset:
```json
[
    "Your custom prompt 1",
    "Your custom prompt 2",
    ...
]
```

### Different Models

Change model in config:
```python
config_dict = {
    "modelname": "gpt2-medium"  # or gpt2-large, TinyLlama, etc.
}
```

---

## 🐛 Troubleshooting

**CUDA out of memory:**
```bash
# Use smaller batch sizes or CPU
python benchmark_all_schedulers.py  # auto-detects device
```

**vLLM connection error:**
```bash
# 1. Start server
cd scheduler && ./start_server.sh

# 2. Test connection
python tests/test_vllm.py

# 3. Check endpoint
curl http://localhost:8000/v1/models
```

**Plot generation fails:**
```bash
# Ensure matplotlib backend
export MPLBACKEND=Agg
python benchmark_plot.py
```

---

## 📚 Core Concepts Summary

### Batching
- **Why?** GPU parallelism - process multiple requests simultaneously
- **Trade-off:** Latency (waiting for batch) vs. Throughput (GPU utilization)

### Time to First Token (TTFT)
- **Definition:** Time until first token is generated
- **Importance:** User experience (streaming responses)
- **Lower is better** for interactive applications

### Throughput
- **Requests/sec:** System capacity
- **Tokens/sec:** Actual generation speed (better metric)
- **Higher is better** for overall performance

### Latency Percentiles
- **P50 (median):** Typical user experience
- **P95:** 95% of requests faster than this
- **P99:** Tail latency (worst-case for most users)

### Memory Management
- **Static (Naive):** Fixed allocation, can waste memory
- **Dynamic:** Adaptive scaling based on pressure
- **PagedAttention (vLLM):** Virtual memory for KV cache (most efficient)

---

## 📖 References

- [vLLM Documentation](https://docs.vllm.ai/)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Transformers Library](https://huggingface.co/docs/transformers)

---

## 🤝 Contributing

To add a new scheduler:
1. Implement in `scheduler/your_scheduler.py`
2. Add interface: `async def add_request(prompt: str) -> str`
3. Update `scheduler_dict` in `benchmark_all_schedulers.py`
4. Run benchmark and generate plots

---

**Built with:** PyTorch • Transformers • vLLM • Matplotlib • AsyncIO
