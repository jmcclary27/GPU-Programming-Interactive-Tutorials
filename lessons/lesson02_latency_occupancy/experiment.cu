// experiment.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
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

int main(int argc, char** argv)
{
    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    const int device_id = 0;
    CHECK_CUDA(cudaSetDevice(device_id));

    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDeviceProperties(&props, device_id));

    // N is a power-of-two so latency_kernel can use bitmasks.
    const int    N     = 1 << 20;  // 1,048,576 elements
    const size_t BYTES = static_cast<size_t>(N) * sizeof(float);

    int iters = 256;
    if (argc > 1) {
        iters = std::atoi(argv[1]);
        if (iters <= 0) iters = 256;
    }

    std::printf("==============================================\n");
    std::printf("Lesson 2 — Latency Hiding & Occupancy (experiment)\n");
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
    // Host data
    // -------------------------------------------------------------------------
    std::vector<float> h_in(N);
    std::vector<float> h_out(N);

    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>((i % 97) * 0.123f);
    }

    // -------------------------------------------------------------------------
    // Device memory
    // -------------------------------------------------------------------------
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, BYTES));
    CHECK_CUDA(cudaMalloc(&d_out, BYTES));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), BYTES, cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // Block-size sweep
    // -------------------------------------------------------------------------
    const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    const int num_configs   = sizeof(block_sizes) / sizeof(block_sizes[0]);

    std::printf("%8s %8s %12s %15s %15s\n",
                "Block", "Grid", "Occ/SM (%)", "Time (ms)", "BW approx (GB/s)");
    std::printf("-------------------------------------------------------------------------------\n");

    for (int c = 0; c < num_configs; ++c) {
        int block = block_sizes[c];

        // Some GPUs do not support 1024 threads per block; skip if so.
        if (block > props.maxThreadsPerBlock) {
            std::printf("%8d %8s %12s %15s %15s  (unsupported block size)\n",
                        block, "-", "-", "-", "-");
            continue;
        }

        int grid = (N + block - 1) / block;

        // Theoretical occupancy (per SM).
        int numBlocksPerSm = 0;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSm, latency_kernel, block, 0));

        int   activeThreadsPerSm = numBlocksPerSm * block;
        float occ = static_cast<float>(activeThreadsPerSm) /
                    static_cast<float>(props.maxThreadsPerMultiProcessor);
        if (occ > 1.0f) occ = 1.0f;

        // Warm-up launch.
        latency_kernel<<<grid, block>>>(d_in, d_out, iters, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // Timed launch.
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        std::printf("Launching kernel...\n");
        std::printf("Grid size:  %d\n", grid);
        std::printf("Block size: %d\n", block);

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
        //   Each thread:
        //     - reads one float per iteration -> iters * 4 bytes
        //     - writes one float at the end  -> 4 bytes
        //
        //   Total bytes ≈ N * (iters + 1) * 4
        double bytes_total = static_cast<double>(N)
                           * static_cast<double>(iters + 1)
                           * sizeof(float);
        double seconds  = ms / 1000.0;
        double gb_per_s = (bytes_total / seconds) / 1e9;

        std::printf("%8d %8d %11.1f%% %15.3f %15.2f\n",
                    block, grid, occ * 100.0f, ms, gb_per_s);
        std::printf("\n");
    }

    // Light sanity check: copy back and compute checksum.
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, BYTES, cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (int i = 0; i < N; ++i) {
        checksum += h_out[i];
    }
    std::printf("Checksum (use to verify runs are consistent): %.6e\n", checksum);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}