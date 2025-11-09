#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloGPUWorld() {
    int threadID = threadIdx.x;
    printf("Hello, GPU World! %d\n", threadID);
}

int main() {
    helloGPUWorld<<<1,32>>>();

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();

    return 0;

}
