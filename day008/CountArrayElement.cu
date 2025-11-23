#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>


__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int cur = 0;
    #pragma unroll 4
    while(i < N) {
        cur += (input[i] == K);
        i += stride;
    }
    atomicAdd(output, cur);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    cudaMemset(output, 0, sizeof(int));
    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}

int main() {
    const int N = 1 << 20;
    const int K = 42;

    // Build host test array
    std::vector<int> h_input(N, 0);

    int positions[10] = {5, 100, 12345, 55555, 77777, 100000, 200000, 300000, 400000, 500000};
    for (int idx : positions) {
        h_input[idx] = K;
    }

    // Allocate GPU memory
    int* d_input;
    int* d_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));

    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    solve(d_input, d_output, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time elapsed: " << milliseconds << " ms\n";

    int h_output = 0;
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Verification
    std::cout << "GPU counted: " << h_output << "\n";

    if (h_output == 10) {
        std::cout << "PASS: correct number of matches.\n";
    } else {
        std::cout << "FAIL: expected 10 but got " << h_output << "\n";
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}