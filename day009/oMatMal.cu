#include <stdio.h>
#include <cuda_runtime.h>


#define N 1024
#define TILE 16



__global__ void matrixMultiplication(const float* A, const float* B, float* C) {
    extern __shared__ float sharedMem[];
    float* tileA = sharedMem;
    float* tileB = sharedMem + blockDim.x * blockDim.y;

    int row = blockDim.y * blockIdx.y * threadIdx.y;
    int col = blockDim.x * blockIdx.x * threadIdx.y;

    float val = 0.0f;

    #pragma unroll 4;
    for (int t = 0; t < N; t += TILE) {

        int a_index_col = t + threadIdx.x;
        if (row < N && a_index_col < N)
            tileA[threadIdx.y * TILE + threadIdx.x] = A[row * N + a_index_col];
        else
            tileA[threadIdx.y * TILE + threadIdx.x] = 0.0f;

        int b_index_row = t + threadIdx.y;
        if (b_index_row < N && col < N)
            tileB[threadIdx.y * TILE + threadIdx.x] = B[b_index_row * N + col];
        else
            tileB[threadIdx.y * TILE + threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            val += tileA[threadIdx.y * TILE + k] *
                     tileB[k * TILE + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = val;
}




int main() {
    // matrix size
    size_t size = N * N * sizeof(float);

    // allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // fill host memory
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    // initialize device memory and allocate space
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // copy host memory to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // block sizes to test
    dim3 blockSizes[] = {dim3(1,1), dim3(1,1), dim3(2,2), dim3(4,4), dim3(8,8), dim3(16,16), dim3(32,32)};
    int numTests = 7;

    // run tests
    printf("===== GPU BENCHMARK =====\n");
    for (int t = 0; t < numTests; t++) {
        dim3 block = blockSizes[t];
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

        size_t sharedMem = 2 * TILE * TILE * sizeof(float);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        printf("\nBlock size: (%d, %d)\n", block.x, block.y);

        cudaEventRecord(start);
        matrixMultiplication<<<grid, block, sharedMem>>>(d_A, d_B, d_C);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("GPU time: %.3f ms\n", ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // free mem
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}