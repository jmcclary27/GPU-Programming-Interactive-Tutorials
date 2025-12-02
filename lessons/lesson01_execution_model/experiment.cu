#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Declare the kernel (defined in kernel.cu)
extern "C" __global__
void vector_add(const float* a, const float* b, float* out, int n);

// Simple CUDA error check helper
static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " -> "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    int n = 1 << 20; // 1 million elements
    size_t bytes = n * sizeof(float);

    // Host vectors
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_out(n, 0.0f);

    // Device pointers
    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    check_cuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    check_cuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    check_cuda(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");

    // Copy host → device
    check_cuda(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy h_a -> d_a");
    check_cuda(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy h_b -> d_b");

    // Kernel launch configuration
    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    std::cout << "Launching kernel…" << std::endl;
    std::cout << "Grid size:  " << gridSize  << std::endl;
    std::cout << "Block size: " << blockSize << std::endl;

    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_out, n);

    // Check for launch errors and sync
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Copy back
    check_cuda(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy d_out -> h_out");

    // Validate
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (h_out[i] != 3.0f) {
            ok = false;
            // Print a few bad values to help debug
            std::cerr << "Mismatch at index " << i
                      << ": got " << h_out[i]
                      << ", expected 3.0" << std::endl;
            break;
        }
    }

    std::cout << (ok ? "SUCCESS: output correct!" : "ERROR: output incorrect!")
              << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return ok ? 0 : 1;
}