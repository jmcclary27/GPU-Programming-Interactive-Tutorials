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
//   1. Find the lines marked with "TODO" in the comments.
//   2. Replace the placeholder code on those lines with your own code.
//   3. Compile and run:
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

extern "C" __global__
void vector_add_exercise(const float* a,
                         const float* b,
                         float* out,
                         int n) {
    /* TODO: compute a global index for this thread using blockIdx.x, blockDim.x, and threadIdx.x
    replace this placeholder with the correct global index expression */
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    /* TODO: use a bounds check so threads with idx outside [0, n) do nothing
    replace this condition so valid threads (within the array) run the body */
    if (idx < n) {
        /* TODO: perform the vector addition using the correct index
        replace this placeholder write with the correct expression */
        out[idx] = a[idx] + b[idx];
    }
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

    /* TODO: compute how many blocks we need so that every element [0, n) is covered
    replace this placeholder with a correct expression using n and blockSize */
    int gridSize = 4;

    std::cout << "Exercise kernel launch" << std::endl;
    std::cout << "  n         = " << n         << " elements" << std::endl;
    std::cout << "  blockSize = " << blockSize << " threads per block" << std::endl;
    std::cout << "  gridSize  = " << gridSize  << " blocks" << std::endl;

    /* TODO: launch the kernel using your gridSize and blockSize and pass all four kernel arguments
    replace the launch configuration and/or arguments as needed */
    vector_add_exercise<<<gridSize, blockSize>>>(d_a, d_b, d_out, n);

    // ************************ DO NOT CHANGE ANYTHING BELOW THIS LINE ************************ //

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