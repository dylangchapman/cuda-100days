#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    #pragma unroll 4
    while(i < N) {
        output[i] = fmaxf(0.0, input[i]);
        i += stride;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize(); 
}
