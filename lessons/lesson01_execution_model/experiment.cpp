#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

extern "C" __global__
void vector_add(const float* a, const float* b, float* out, int n);

int main() {
    int n = 1 << 20; // 1 million elements
    size_t bytes = n * sizeof(float);

    // Host vectors
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_out(n);

    // Device pointers
    float *d_a, *d_b, *d_out;
    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_out, bytes);

    // Copy host → device
    hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice);

    // Kernel launch configuration
    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    std::cout << "Launching kernel…" << std::endl;
    std::cout << "Grid size:  " << gridSize  << std::endl;
    std::cout << "Block size: " << blockSize << std::endl;

    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_out, n);
    hipDeviceSynchronize();

    // Copy back
    hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost);

    // Validate
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (h_out[i] != 3.0f) {
            ok = false;
            break;
        }
    }

    std::cout << (ok ? "SUCCESS: output correct!" : "ERROR: output incorrect!") 
              << std::endl;

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_out);

    return 0;
}