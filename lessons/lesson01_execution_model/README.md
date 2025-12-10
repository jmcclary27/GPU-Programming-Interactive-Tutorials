# Lesson 1 — CUDA Execution Model  
### Threads, Blocks, Grids, and Global Indexing

Welcome to **Lesson 1** of the *GPU Programming Interactive Tutorials*.  
This lesson introduces the *execution model* of CUDA, which is the foundation for every GPU program you will write later in this course.

---

## 📄 Learning Material

### **Lesson 1 Concepts Document**  
A written explanation of all Lesson 1 topics:  
👉 **[Click here to open the Lesson 1 Document](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)**  

### **Lesson 1 Video Walkthrough**  
A full visual explanation of threads, blocks, grids, and indexing:  
👉 **[Watch the Lesson 1 Video](https://www.youtube.com/watch?v=cRY5utouJzQ)**  

---

# 🚀 What You Will Learn in Lesson 1

This lesson teaches the most essential CUDA concepts:

### **1. Threads, Blocks, and Grids**
- The GPU runs thousands of *threads*.  
- Threads are grouped into *blocks*.  
- Blocks form a *grid*.

### **2. Global Thread Indexing**
Every thread must compute a unique global index:

global_idx = blockIdx.x * blockDim.x + threadIdx.x

### **3. Bounds Checking**
Threads often outnumber the data size, so you must always protect memory:

if (idx < n) { ... }

### **4. Simple Elementwise Operations**
You learn the canonical “Hello World of CUDA”:

out[i] = a[i] + b[i]

These concepts appear in **every CUDA kernel you will ever write**, so mastering them here is critical.

---

# 📘 Files Included in This Lesson

This lesson contains **four** CUDA programs:

1. **print_demo.cu** — shows thread-to-index mapping  
2. **kernel.cu** — the correct reference vector-add kernel  
3. **experiment.cu** — large-scale vector-add using the reference kernel  
4. **lesson1_exercise.cu** — a fill-in-the-blanks exercise that requires ALL Lesson 1 concepts

Each file builds toward understanding the core execution model.

---

# 🖨️ Demo: `print_demo.cu`

This is a tiny, visual demonstration of how CUDA assigns work.

It launches a grid of threads and prints:

block B, thread T --> global index I

This helps you *see* the mapping between:
- `blockIdx.x`
- `threadIdx.x`
- and the computed global index.

## How To Compile

(Might need to change compile command depending on device being used)
nvcc print_demo.cu -o print_demo
./print_demo

### **How this demo works**
- You set `n`, `blockSize`, and `gridSize`.  
- The kernel computes a global index.  
- If the index is in bounds, it prints the mapping.  
- Running it with different block sizes helps you understand how threads cover data.

This file does **no math**—it simply visualizes the execution model.

---

# 🧪 Demo: `experiment.cu` + `kernel.cu`

These two files work together to show a **real**, fully functioning CUDA kernel.

## How to compile

(Might need to change compile command depending on device being used)
nvcc experiment.cu kernel.cu -o experiment
./experiment

### `kernel.cu`
Defines the correct version of the vector-add kernel:
- Computes global index  
- Checks bounds  
- Writes `out[idx] = a[idx] + b[idx]`

### `experiment.cu`
Runs the real kernel at scale:
- Allocates 1 million floats  
- Copies host data → device  
- Launches the kernel with:

blockSize = 256
gridSize = (n + blockSize - 1) / blockSize
- Copies results back  
- Validates correctness  
- Prints whether the GPU output is correct

### **What this demo teaches**
- Realistic block and grid sizing  
- Kernel launch syntax  
- GPU/CPU memory transfers  
- Error checking  
- Correct vector add behavior  

This is the *reference program* that the exercise is modeled after.

---

# 📝 Exercise: `lesson1_exercise.cu`

This is the **core hands-on challenge for Lesson 1**.

You are given a full CUDA program **with missing pieces you must fill in**.

### ❗ What You Must Implement Yourself
Inside the kernel, you must write all logic for:
1. Computing the global index  
2. Performing bounds checking  
3. Writing the vector addition result  
4. Configuring the grid-size correctly in `main`  
5. Launching the kernel correctly in `main`

All `TODO` sections are the exact spots where you must write code.

### 💡 Why this exercise is important  
To make the program work, you must demonstrate that you understand:
- Threads, blocks, and grids  
- Global thread indexing  
- Ceiling-division grid sizing  
- Bounds checking  
- Elementwise vector addition  
- Correct kernel launch syntax  

The program prints `"SUCCESS"` only if **every part of your implementation is correct**.

### ▶️ How to run the exercise

(Might need to change compile command depending on device being used)
nvcc lesson1_exercise.cu -o lesson1_exercise
./lesson1_exercise

If anything is incorrect:
- The output will be wrong, OR  
- The kernel will not launch properly, OR  
- Bounds checking will fail, OR  
- You will get `"ERROR: output incorrect!"`

You are expected to fix it using Lesson 1 skills.

---

# 🧩 How These Files Work Together

| File | Purpose |
|------|---------|
| **print_demo.cu** | Shows thread → global index mapping visually |
| **kernel.cu** | The correct vector-add kernel used in the large experiment |
| **experiment.cu** | Validates how the real kernel behaves with big data |
| **lesson1_exercise.cu** | Requires you to write your own kernel + launch logic |

By the end of Lesson 1, you will know **exactly how threads map onto data**, and you will have written your **first real CUDA kernel**.

---

# ✔️ Next Step

Move to **Lesson 1: The Exercise** and fill in the missing code.  
Once you get `"SUCCESS"`, you’re ready for Lesson 2.

---

# Exercise Answers (Don't Read Before Attempting Exercise.cu)

```cpp
int idx = threadIdx.x + blockIdx.x * blockDim.x;



if (idx < n) {
    out[idx] = a[idx] + b[idx];
}



int gridSize = (n + blockSize - 1) / blockSize;



vector_add_exercise<<<gridSize, blockSize>>>(d_a, d_b, d_out, n);
```