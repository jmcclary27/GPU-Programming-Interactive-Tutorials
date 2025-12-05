// print_demo.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "kernel.cuh"

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                         __FILE__, __LINE__, cudaGetErrorString(err_));   \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

int main()
{
    const int device_id = 0;
    CHECK_CUDA(cudaSetDevice(device_id));

    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDeviceProperties(&props, device_id));

    const int N     = 1 << 20; // 1,048,576 elements
    const int iters = 256;     // work per thread

    std::printf("==============================================\n");
    std::printf("Lesson 2 — Latency Hiding & Occupancy (print_demo)\n");
    std::printf("Device: %s\n", props.name);
    std::printf("SMs: %d, max threads / SM: %d, warp size: %d\n",
                props.multiProcessorCount,
                props.maxThreadsPerMultiProcessor,
                props.warpSize);
    std::printf("Problem size: N = %d\n", N);
    std::printf("Inner loop iters per thread: %d\n", iters);
    std::printf("==============================================\n\n");

    // Show a single example launch like in Lesson 1.
    int block = 256;
    if (block > props.maxThreadsPerBlock) {
        block = props.maxThreadsPerBlock;
    }
    int grid = (N + block - 1) / block;

    std::printf("Launching kernel...\n");
    std::printf("Grid size:  %d\n", grid);
    std::printf("Block size: %d\n\n", block);

    // Compute theoretical occupancy for this launch configuration.
    int numBlocksPerSm = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, latency_kernel, block, 0));

    int activeThreadsPerSm = numBlocksPerSm * block;
    float occ = static_cast<float>(activeThreadsPerSm) /
                static_cast<float>(props.maxThreadsPerMultiProcessor);
    if (occ > 1.0f) occ = 1.0f;

    std::printf("For block size %d:\n", block);
    std::printf("  Active blocks per SM:  %d\n", numBlocksPerSm);
    std::printf("  Active threads per SM: %d\n", activeThreadsPerSm);
    std::printf("  Theoretical occupancy: %.1f%%\n\n", occ * 100.0f);

    // Show an occupancy table for a range of block sizes.
    const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    const int num_configs   = sizeof(block_sizes) / sizeof(block_sizes[0]);

    std::printf("Occupancy vs block size (no timing, just theory):\n");
    std::printf("%8s %12s %16s\n", "Block", "Blocks/SM", "Occ/SM (%)");
    std::printf("------------------------------------------\n");

    for (int i = 0; i < num_configs; ++i) {
        int b = block_sizes[i];

        if (b > props.maxThreadsPerBlock) {
            std::printf("%8d %12s %16s  (unsupported)\n",
                        b, "-", "-");
            continue;
        }

        int blocksPerSm = 0;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSm, latency_kernel, b, 0));

        int threadsPerSm = blocksPerSm * b;
        float occ_b = static_cast<float>(threadsPerSm) /
                      static_cast<float>(props.maxThreadsPerMultiProcessor);
        if (occ_b > 1.0f) occ_b = 1.0f;

        std::printf("%8d %12d %15.1f%%\n",
                    b, blocksPerSm, occ_b * 100.0f);
    }

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}