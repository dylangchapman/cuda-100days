#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x < (N / 2)) {
        int len = N - 1;
        float temp = input[x];
        input[x] = input[len - x];
        input[len - x] = temp;
    }

}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
