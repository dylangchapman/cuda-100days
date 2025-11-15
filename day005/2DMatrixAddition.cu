#include <stdio.h>
#include <cuda_runtime.h>


#define ROWS 1000
#define COLS 1000


__global__ void 2DMatrixAdd(const float *A, const float *B, float *C, int N, int M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < M) {
        int idx = row * M + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t size = ROWS * COLS * sizeof(float);

    const dim3 blockSizes[] = {dim3(1,1), dim3(32,32), dim3(16,16), dim3(8,8)}; // (1,1) kernel is used as a warmup, else 32x32 will be have overhead time
    const int num_tests = 4;

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for(int i = 0; i < ROWS * COLS; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    for(int t = 0; t < num_tests; t++) {
        dim3 blockSize = blockSizes[t];
        dim3 gridSize((COLS + blockSize.x - 1) / blockSize.x, (ROWS + blockSize.y - 1) / blockSize.y);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        2DMatrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, ROWS, COLS);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Block Size: (%d, %d), Time: %f ms\n", blockSize.x, blockSize.y, milliseconds);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}