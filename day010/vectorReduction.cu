#include <cuda_runtime.h>
#include <math.h>

__global__ void naiveVectorReduction(const float* input, float* partial, float* scratch, int N) {
    int tid   = threadIdx.x;
    int gid   = blockIdx.x * blockDim.x + tid;
    // get block starting index for reduction
    int blockStart = blockIdx.x * blockDim.x;

    if(gid < N) {
        scratch[blockStart + tid] = input[gid];
    } 
    else {
        scratch[blockStart + tid] = 0.0f;
    }
    __syncthreads();

    // reduce within block
    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        if(tid < stride) {
            scratch[blockStart + tid] += scratch[blockStart + tid + stride];
        }
        __syncthreads();
    }

    // write partial sum at start of block
    if(tid == 0) {
        partial[blockIdx.x] = scratch[blockStart];
    }
}

__global__ void sharedMemVectorReduction(const float* input, float* partial, int N) {
    extern __shared__ float sharedMem[];
    int tid   = threadIdx.x;
    int gid   = blockIdx.x * blockDim.x + tid;

    // load input into shared memory
    if(gid < N) {
        sharedMem[tid] = input[gid];
    } 
    else {
        sharedMem[tid] = 0.0f;
    }
    __syncthreads();

    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1){
        if(tid < stride) {
            sharedMem[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

__global__ void warpOptimizedVectorReduction(const float* input, float* partial, int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float val = (gid < N) ? input[gid] : 0.0f;

    unsigned mask = 0xffffffff;

    // 32 -> 16 -> 8 -> 4 -> 2 -> 1
    val += __shfl_down_sync(mask, val, 16);
    val += __shfl_down_sync(mask, val, 8);
    val += __shfl_down_sync(mask, val, 4);
    val += __shfl_down_sync(mask, val, 2);
    val += __shfl_down_sync(mask, val, 1);

    int lane   = tid & 31; 
    int warpId = tid >> 5; 

    __shared__ float warpSums[32];
    if (lane == 0) {
        warpSums[warpId] = val;
    }
    __syncthreads();
    float blockVal = 0.0f;
    if (warpId == 0) {
        // lane < number of warps?
        if (lane < (blockDim.x >> 5)) {
            blockVal = warpSums[lane];
        }

        blockVal += __shfl_down_sync(mask, blockVal, 16);
        blockVal += __shfl_down_sync(mask, blockVal, 8);
        blockVal += __shfl_down_sync(mask, blockVal, 4);
        blockVal += __shfl_down_sync(mask, blockVal, 2);
        blockVal += __shfl_down_sync(mask, blockVal, 1);

        if (lane == 0) {
            partial[blockIdx.x] = blockVal;
        }
    }
}

int main(int argc, char** argv)  {
    // allows me to input any arb 2^x size from command line
    int N = 1024;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    int blockSize = 1024;
    int numBlocks = N / blockSize;

    printf("N = %d, blocks = %d, blockSize = %d\n", N, numBlocks, blockSize);

    // allocate host mem
    float* h_In       = (float*)malloc(N * sizeof(float));
    float* h_Partials = (float*)malloc(numBlocks * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_In[i] = 1.0f;
    }

    // allocate device mem
    float *d_In, *d_Partials, *d_Scratch;

    cudaMalloc(&d_In,      N * sizeof(float));
    cudaMalloc(&d_Partials, numBlocks * sizeof(float));
    cudaMalloc(&d_Scratch,  numBlocks * blockSize * sizeof(float));

    cudaMemcpy(d_In, h_In, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // naive
    cudaEventRecord(start);

    naiveVectorReduction<<<numBlocks, blockSize>>>(d_In, d_Partials, d_Scratch, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t_naive = 0;
    cudaEventElapsedTime(&t_naive, start, stop);

    cudaMemcpy(h_Partials, d_Partials, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    double sum_naive = 0;
    for (int i = 0; i < numBlocks; i++) sum_naive += h_Partials[i];

    printf("\nNaive Reduction:\n");
    printf("  Time = %.3f ms\n", t_naive);
    printf("  Result = %.1f (expected %.1f)\n", sum_naive, (double)N);

    // shared
    cudaEventRecord(start);

    sharedMemVectorReduction<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_In, d_Partials, N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t_shared = 0;
    cudaEventElapsedTime(&t_shared, start, stop);

    cudaMemcpy(h_Partials, d_Partials, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    double sum_shared = 0;
    for (int i = 0; i < numBlocks; i++) sum_shared += h_Partials[i];

    printf("\nShared Memory Reduction:\n");
    printf("  Time = %.3f ms\n", t_shared);
    printf("  Result = %.1f (expected %.1f)\n", sum_shared, (double)N);

    // warps
    cudaEventRecord(start);

    warpOptimizedVectorReduction<<<numBlocks, blockSize>>>(d_In, d_Partials, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t_shuffle = 0;
    cudaEventElapsedTime(&t_shuffle, start, stop);

    cudaMemcpy(h_Partials, d_Partials, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    double sum_shuffle = 0;
    for (int i = 0; i < numBlocks; i++) sum_shuffle += h_Partials[i];

    printf("\nWarp Shuffle Reduction:\n");
    printf("  Time = %.3f ms\n", t_shuffle);
    printf("  Result = %.1f (expected %.1f)\n", sum_shuffle, (double)N);

    // frees
    cudaFree(d_In);
    cudaFree(d_Partials);
    cudaFree(d_Scratch);
    free(h_In);
    free(h_Partials);

    printf("\nDone.\n");
    return 0;
}
