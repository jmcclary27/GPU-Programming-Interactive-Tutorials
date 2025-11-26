#include <hip/hip_runtime.h>
#include <stdio.h>

extern "C" __global__
void print_thread_mapping(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        printf("block %d, thread %d --> global index %d\n",
               blockIdx.x, threadIdx.x, idx);
    }
}
