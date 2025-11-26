# GPU Programming Interactive Tutorials
A hands-on, experiment-driven introduction to GPU kernels, memory hierarchy, and performance optimization — using HIP.

This repository is an **interactive learning environment** for anyone who wants to understand how GPU compute programming *really* works.  
Instead of reading theory, you will **run kernels**, **measure performance**, **visualize results**, and **discover GPU architecture concepts by experimentation**.

---

## What You’ll Learn

By completing the 12 interactive lessons in this repo, you will understand:

### GPU Execution & Threading
- Threads, blocks, grids  
- Wavefronts/warps and occupancy  
- How GPUs hide latency with massive parallelism  

### GPU Memory Hierarchy
- Global memory, caches, shared memory  
- Memory coalescing and strided accesses  
- Latency vs bandwidth bottlenecks  
- Shared memory bank conflicts  

### Performance Behavior & Optimization
- Launch parameters (block size, tile size)  
- Divergence and branch coherence  
- Tiling and data reuse  
- Compute-bound vs memory-bound workloads  
- Overlapping compute and memory transfers with streams  

### Benchmarking & Profiling
- Accurate timing with HIP events  
- Running parameter sweeps  
- Visualizing bandwidth, GFLOPs, and latency  
- Interpreting performance curves  

These topics reflect the fundamentals used by AMD, NVIDIA, and Intel GPU engineers.

---

## Lessons Overview

Each lesson includes:
- A **concept explanation**  
- A **runnable HIP kernel**  
- An **experiment** (parameter sweep)  
- A **plot or visualization**  
- A **discussion of results**  

### **Lesson 1 — GPU Execution Model Basics**  
Threads, blocks, grids, indexing (vector add).

### **Lesson 2 — Latency Hiding & Occupancy**  
Block size sweep, undersaturation vs optimal occupancy.

### **Lesson 3 — Memory Hierarchy**  
Global memory, caching, memory latency.

### **Lesson 4 — Memory Coalescing**  
Coalesced vs non-coalesced accesses.

### **Lesson 5 — Shared Memory & Bank Conflicts**  
Why shared memory is fast, how conflicts occur.

### **Lesson 6 — Divergence & Branching**  
Random vs coherent branching and its effect on throughput.

### **Lesson 7 — Tiling & Data Reuse**  
Shared-memory tiled matrix multiply.

### **Lesson 8 — Compute-Bound vs Memory-Bound Workloads**  
Arithmetic intensity and roofline intuition.

### **Lesson 9 — Kernel Launch Configuration Tuning**  
Block/grid size sweeps and occupancy tradeoffs.

### **Lesson 10 — Asynchronous Operations & Streams**  
Copy/compute overlap with HIP streams.

### **Lesson 11 — Profiling with HIP Events**  
Consistent timing and metric collection.

### **Lesson 12 — Putting It All Together**  
Mini-projects focused on tuning and performance analysis.

---

## Repository Structure

```

gpu-programming-interactive-tutorials/
│
├── lessons/
│   ├── lesson01_execution_model/
│   │   ├── kernel.cu
│   │   ├── experiment.cpp
│   │   └── README.md
│   │
│   ├── lesson02_occupancy/
│   ├── lesson03_memory_hierarchy/
│   ├── lesson04_coalescing/
│   ├── lesson05_shared_memory/
│   ├── lesson06_divergence/
│   ├── lesson07_tiling/
│   ├── lesson08_compute_vs_memory_bound/
│   ├── lesson09_launch_config/
│   ├── lesson10_async_streams/
│   ├── lesson11_profiling/
│   └── lesson12_projects/
│
├── common/
│   ├── utils.h
│   ├── timing.h
│   ├── plotting/
│   │   └── (Python or JS plotting helpers)
│   └── data/
│
├── python/
│   ├── analyze_results.ipynb
│   └── plot_results.py
│
├── results/
│   ├── example_plots/
│   └── csv/
│
└── README.md

```

## Requirements

### **Software**
- HIP / ROCm toolchain (or HIP CPU backend)  
- CMake or hipcc
- Python 3.10+  
- Pandas  
- Matplotlib or Plotly for visualizations  

### **Hardware**
You **do not need an AMD GPU**.  
All fundamental experiments run using HIP’s CPU backend or any supported GPU.