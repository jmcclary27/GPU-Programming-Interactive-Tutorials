# GPU Programming Interactive Tutorials
A hands-on, experiment-driven introduction to GPU kernels, memory hierarchy, and performance optimization — using HIP.

This repository is an **interactive learning environment** for anyone who wants to understand how GPU compute programming *really* works.  
Instead of reading theory, you will **run kernels**, **measure performance**, **visualize results**, and **discover GPU architecture concepts by experimentation**.

---

## **Hardware**
You **do not need an AMD GPU**.
These tutorials use AMD's HIP, however any supported GPU or just a CPU backend will still work.

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

## Requirements
- Run the following command in the root of the repo to install the required packages: pip install -r requirements.txt
- No GPU is necessary to run the experiments

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

---

## Repository Structure

```

gpu-programming-interactive-tutorials/
│
├── lessons/
│   └── lesson01_execution_model/
│       ├── kernel.cu
│       ├── experiment.cpp
│       └── README.md
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