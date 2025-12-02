// lesson1_exercise.cu
//
// Exercise: fill in the missing pieces to make this program work.
//
// Concepts you should use from Lesson 1:
//   - Threads, blocks, and grids
//   - Global indexing with blockIdx.x, blockDim.x, and threadIdx.x
//   - Bounds checking (idx < n)
//   - Simple vector add: out[i] = a[i] + b[i]
//
// Instructions:
//   1. Find the sections marked "TODO".
//   2. Uncomment those lines and replace the "..." with real code.
//   3. When you think you're done, compile and run:
//
//        nvcc lesson1_exercise.cu -o lesson1_exercise
//        ./lesson1_exercise
//
//   4. If you see "SUCCESS", your code is correct. Otherwise, fix your kernel
//      or launch configuration until the check passes.

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <stdio.h>

// TODO: implement this kernel using the concepts from Lesson 1
extern "C" __global__
void vector_add_exercise(const float* a,
                         const float* b,
                         float* out,
                         int n) {
    // Step 1: compute a global index for this thread
    //   Use blockIdx.x, blockDim.x, and threadIdx.x
    //
    // int idx = ...;  // TODO: uncomment and replace "..." with the correct expression

    // Step 2: check that idx is inside the bounds of the array
    //   Only access a[idx], b[idx], out[idx] if idx < n
    //
    // if (idx < n) {
    //     // Step 3: perform the vector addition
    //     //   out[idx] = a[idx] + b[idx];
    //     //
    //     // out[idx] = ...;  // TODO: uncomment and complete this line
    //
    //     // (Optional, but recommended for learning)
    //     // Print the mapping from (block, thread) to the global index.
    //     // printf("block %d, thread %d --> global index %d\n",
    //     //        blockIdx.x, threadIdx.x, idx);
    // }
}

int main() {
    // Use a small size so it's easy to print and reason about.
    int n = 32;
    size_t bytes = n * sizeof(float);

    // Host vectors
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_out(n, 0.0f);

    // Device pointers
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_out = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, bytes);

    // Copy host → device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    // Choose how many threads per block to use.
    // Try changing this value later (e.g., 4, 8, 16) and see what happens.
    int blockSize = 8;

    // TODO: compute how many blocks we need to cover n elements
    //   Hint: use the "ceiling division" pattern you saw in the example:
    //     gridSize = (n + blockSize - 1) / blockSize;
    //
    // int gridSize = ...;  // TODO: uncomment and replace "..." with the correct expression

    std::cout << "Exercise kernel launch" << std::endl;
    std::cout << "  n         = " << n         << " elements" << std::endl;
    std::cout << "  blockSize = " << blockSize << " threads per block" << std::endl;
    // std::cout << "  gridSize  = " << gridSize  << " blocks" << std::endl;  // TODO: uncomment after you define gridSize

    // TODO: launch the kernel with your gridSize and blockSize
    //   vector_add_exercise<<< /* gridSize */, /* blockSize */ >>>( /* arguments */ );
    //
    // vector_add_exercise<<< ..., ... >>>( ..., ..., ..., ... );  // TODO: uncomment and fill in

    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    // Check correctness: every element should be 3.0f
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (h_out[i] != 3.0f) {
            ok = false;
            break;
        }
    }

    std::cout << (ok ? "SUCCESS: output correct!"
                     : "ERROR: output incorrect, fix your kernel or launch")
              << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return ok ? 0 : 1;
}