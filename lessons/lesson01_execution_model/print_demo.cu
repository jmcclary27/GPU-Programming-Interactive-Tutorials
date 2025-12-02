#include <cstdio>
#include <cuda_runtime.h>

__global__ void print_thread_mapping(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        printf("block %d, thread %d --> global index %d\n",
               blockIdx.x, threadIdx.x, idx);
    }
}

int main() {
    int n = 16;
    int blockSize = 4;
    int gridSize  = (n + blockSize - 1) / blockSize;

    printf("print_thread_mapping demo\n");
    printf("  n         = %d\n", n);
    printf("  blockSize = %d threads per block\n", blockSize);
    printf("  gridSize  = %d blocks\n\n", gridSize);

    print_thread_mapping<<<gridSize, blockSize>>>(n);
    cudaDeviceSynchronize();

    return 0;
}
