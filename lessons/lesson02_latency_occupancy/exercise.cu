// Lesson 2 — Latency Hiding & Occupancy
// Block-size sweep, undersaturation vs optimal occupancy.
//
// This experiment:
//
// 1. Allocates a big array on the GPU.
// 2. Launches a latency-heavy kernel for several block sizes.
// 3. For each block size it prints:
//      - Grid size
//      - Theoretical occupancy per SM
//      - Kernel runtime (ms)
//      - Approximate global-memory bandwidth (GB/s)
//
// Usage:
//   nvcc -O3 latency_occupancy.cu -o latency_occupancy
//   ./latency_occupancy            # default iters = 256
//   ./latency_occupancy 1024       # heavier kernel, more latency to hide

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err_));   \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

// A simple integer hash to create a pseudo-random access pattern.
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

// -----------------------------------------------------------------------------
// Kernel
// -----------------------------------------------------------------------------
//
// This kernel is intentionally *latency-heavy*:
//
// - Each thread:
//    * walks through global memory with a hashed index pattern
//    * performs 'iters' loads + a few FMAs
//
// - More active warps per SM  => better ability to "hide" the memory latency.
//
__global__ void latency_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               int iters,
                               int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Use a per-thread seed so different threads touch different addresses.
    int mask = n - 1; // n is a power of two.
    int idx = tid & mask;

    float acc = 0.0f;

    // Main latency-heavy loop
    for (int i = 0; i < iters; ++i) {
        int j = hash_index(idx + i, mask);
        float val = in[j];
        // Some arithmetic to keep the FP units busy.
        acc = acc * 1.0001f + val;
    }

    out[tid] = acc;
}

// -----------------------------------------------------------------------------
// Experiment driver
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // Configuration knobs
    // -------------------------------------------------------------------------
    const int device_id = 0;
    CHECK_CUDA(cudaSetDevice(device_id));

    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDeviceProperties(&props, device_id));

    // We pick N as a power-of-two so we can use bitmasks cheaply.
    const int N = 1 << 20;           // 1,048,576 elements
    const size_t BYTES = N * sizeof(float);

    // How much work each thread does in the inner loop (latency to hide).
    int iters = 256;
    if (argc > 1) {
        iters = std::atoi(argv[1]);
        if (iters <= 0) iters = 256;
    }

    std::printf("==============================================\n");
    std::printf("Lesson 2 — Latency Hiding & Occupancy\n");
    std::printf("Device: %s\n", props.name);
    std::printf("SMs: %d, max threads / SM: %d, warp size: %d\n",
                props.multiProcessorCount,
                props.maxThreadsPerMultiProcessor,
                props.warpSize);
    std::printf("Problem size: N = %d (%.1f MiB)\n",
                N, BYTES / (1024.0 * 1024.0));
    std::printf("Kernel iters per thread: %d\n", iters);
    std::printf("==============================================\n\n");

    // -------------------------------------------------------------------------
    // Allocate & initialize host data
    // -------------------------------------------------------------------------
    std::vector<float> h_in(N);
    std::vector<float> h_out(N);

    // Fill with some non-trivial pattern.
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>((i % 97) * 0.123f);
    }

    // -------------------------------------------------------------------------
    // Allocate device memory
    // -------------------------------------------------------------------------
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, BYTES));
    CHECK_CUDA(cudaMalloc(&d_out, BYTES));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), BYTES, cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // Block-size sweep
    // -------------------------------------------------------------------------
    const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    const int num_configs = sizeof(block_sizes) / sizeof(block_sizes[0]);

    std::printf("%8s %8s %12s %15s %15s\n",
                "Block", "Grid", "Occ/SM (%)", "Time (ms)", "BW approx (GB/s)");
    std::printf("-------------------------------------------------------------------------------\n");

    for (int c = 0; c < num_configs; ++c) {
        int block = block_sizes[c];

        // Some GPUs do not support 1024 threads per block; clamp if needed.
        if (block > props.maxThreadsPerBlock) {
            std::printf("%8d %8s %12s %15s %15s (unsupported block size)\n",
                        block, "-", "-", "-", "-");
            continue;
        }

        int grid = (N + block - 1) / block;

        // Theoretical occupancy (per SM) from CUDA runtime.
        int numBlocksPerSm = 0;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSm, latency_kernel, block, 0));

        int activeThreadsPerSm = numBlocksPerSm * block;
        float occ = (float)activeThreadsPerSm /
                    (float)props.maxThreadsPerMultiProcessor;
        if (occ > 1.0f) occ = 1.0f; // Clamp in case of rounding.

        // Warm-up launch (helps reduce first-run overhead).
        latency_kernel<<<grid, block>>>(d_in, d_out, iters, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // Timed launch
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        latency_kernel<<<grid, block>>>(d_in, d_out, iters, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        // Approximate memory traffic:
        //
        // Each thread:
        //   - reads one float per iteration  -> iters * 4 bytes
        //   - writes one float at the end   -> 4 bytes
        //
        // Total bytes ≈ N * (iters + 1) * 4
        double bytes_total = (double)N * (double)(iters + 1) * sizeof(float);
        double seconds = ms / 1000.0;
        double gb_per_s = (bytes_total / seconds) / 1e9;

        std::printf("%8d %8d %11.1f%% %15.3f %15.2f\n",
                    block, grid, occ * 100.0f, ms, gb_per_s);
    }

    // Optionally copy back and do a light sanity check to ensure kernel ran.
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, BYTES, cudaMemcpyDeviceToHost));

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) {
        checksum += h_out[i];
    }
    std::printf("\nChecksum (ignore absolute value, just compare across runs): %.6e\n",
                checksum);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}