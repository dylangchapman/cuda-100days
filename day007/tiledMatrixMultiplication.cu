#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>

#define N 1024
#define TILE 16

void tiledMatrixMultiplicationHost(const float* A, const float* B, float* C) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int t = 0; t < N; t += TILE) {
                // Tiling
                float tile_a[TILE][TILE];
                float tile_b[TILE][TILE];

                for (int i = 0; i < TILE; i++) {
                    for (int j = 0; j < TILE; j++) {

                        int a_row = row;
                        int a_col = t + j;

                        if (a_row < N && a_col < N)
                            tile_a[i][j] = A[a_row * N + a_col];
                        else
                            tile_a[i][j] = 0.0f;

                        int b_row = t + i;
                        int b_col = col;

                        if (b_row < N && b_col < N)
                            tile_b[i][j] = B[b_row * N + b_col];
                        else
                            tile_b[i][j] = 0.0f;
                    }
                }

                for (int k = 0; k < TILE; k++) {
                    sum += tile_a[0][k] * tile_b[k][0];
                }
            }
            C[row * N + col] = sum;
        }
    }
}


__global__ void tiledMatrixMultiplicationDevice(const float* A, const float* B, float* C) {
    extern __shared__ float shared[];

    float* tileA = shared;
    float* tileB = shared + TILE * TILE;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;

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
            value += tileA[threadIdx.y * TILE + k] *
                     tileB[k * TILE + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = value;
}

int main() {
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 blockSizes[] = { dim3(1,1), dim3(16,16), dim3(32,32) };
    int numTests = 3;

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
        tiledMatrixMultiplicationDevice<<<grid, block, sharedMem>>>(d_A, d_B, d_C);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("GPU time: %.3f ms\n", ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("\n===== CPU BENCHMARK =====\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    tiledMatrixMultiplicationHost(h_A, h_B, h_C);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    long cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();

    printf("CPU time: %ld ms\n", cpu_ms);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
