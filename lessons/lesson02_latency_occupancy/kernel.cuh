// latency_kernel.cuh
#pragma once
#include <cuda_runtime.h>

// Simple integer hash to create a pseudo-random access pattern.
// n is a power of 2, so we can use a mask instead of modulo.
__device__ __forceinline__ int hash_index(int x, int mask) {
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return x & mask;
}

// Latency-heavy kernel.
__global__ void latency_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               int iters,
                               int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Use a per-thread seed so different threads touch different addresses.
    int mask = n - 1; // n should be a power of two.
    int idx = tid & mask;

    float acc = 0.0f;

    for (int i = 0; i < iters; ++i) {
        int j = hash_index(idx + i, mask);
        float val = in[j];
        // Some arithmetic to keep FP units busy.
        acc = acc * 1.0001f + val;
    }

    out[tid] = acc;
}