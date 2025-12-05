# Lesson 2 — Latency Hiding & Occupancy (Exercises)

This lesson explores how **block size** affects:

- Theoretical **occupancy**
- The GPU’s ability to **hide memory latency**
- Overall **runtime** and **memory bandwidth**

You will use:

- `print_demo` — quick visualization of occupancy vs block size  
- `lesson02_experiment` — full timing and performance sweep  

---

## Part 1 — Warm-up: Occupancy vs Block Size

1. Build and run the demo:

   ```bash
   nvcc -O2 print_demo.cu -o print_demo
   ./print_demo

2. Copy the occupancy table into your notes. For each supported block size, record:

- Block size
- Blocks per SM
- Theoretical occupancy (%)

3. Answer the following:

- For very small block sizes (e.g., 32), is occupancy high or low? Why?
- At what block size does occupancy first plateau or approach 100%?
- Does increasing block size always increase occupancy? Explain.

## Part 2 - Block Size Sweep: Latency Hiding in Practice

1. Build the experiment:

    ```bash
    nvcc -O3 experiment.cu -o lesson02_experiment

2. Run the experiment:

    ```bash
    ./lesson02_experiment

3. For each block size in the output, record:

- Block size
- Grid size
- Occupancy (% per SM)
- Time (ms)
- Approximate bandwidth (GB/s)

4. Identify:

- Which block size gives the fastest runtime
- Which gives the highest bandwidth
- Which has the highest occupancy

5. Discussion questions:

- For small block sizes (low occupancy), what trends do you see in runtime and bandwidth?
- Around the “good” block sizes (typically 128–256), what changes do you see?
- Does the block size with the highest occupancy always provide the best performance? Why or why not?
- Define undersaturation in your own words using your experiment results.

## Part 3 - Changing Per-Thread Latency

Here you change the inner loop iteration count (iters) to increase or decrease the latency pressure.

Run the experiment three times:

1. Low latency:

    ```bash
    ./lesson02_experiment 32

2. Medium latency (default):

    ```bash
    ./lesson02_experiment 256

3. High latency:

    ```bash
    ./lesson02_experiment 1024

4. Compare across the three runs.

5. Answer:

- When iters is small, how sensitive is performance to block size?
- When iters is large, does the “best” block size change? Why or why not?
- As latency increases, do you see more benefit from higher occupancy? Explain using your numerical results.

## Learning Outcomes

- By completing this lesson, you should be able to:
- Explain how block size influences GPU occupancy
- Identify when a kernel is undersaturated
- Predict how occupancy impacts latency hiding
- Select effective launch configurations for latency-heavy kernels