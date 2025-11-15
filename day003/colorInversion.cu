#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    
    if(i < (width * height * 4)) {
        image[i] = 255 - image[i];       // R
        image[i+1] = 255 - image[i+1];   // G
        image[i+2] = 255 - image[i+2];   // B
        image[i+3] = image[i+3];         // ALPHA (unchanged)
    }

}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}