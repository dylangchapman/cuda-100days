# 100 Days of Writing CUDA Kernels

- Inspired by julienokumu
- Cannot run CUDA on my machine, so will be using Kaggle/Colab free GPUs
- This will be a long process of getting more familiar with C++, CUDA, and Linear Algebra

## Day 0: Hello GPU World Kernel

### Resources:
- Read chapter 1 of Programming Massively Parallel Processors
- GeeksForGeeks C++ references
- julienokumu github references

### Learnings:
- GPU computation speed is expected to continue accelerating (reading)
- Threading: 
    ``` helloGPUWorld<<<1,32>>>(); ```
    Creates threads 0-31

- thread IDs:
    ``` int threadID = threadIdx.x; ``` 
    gets the ID of the particular thread executing some portion of code (in this case, threads 0-31 of the prints)

- cudaDeviceSynchronize: Current understanding is that this functions works as a lock to keep the system from moving on while Kernel threads are still running

### Challenges:
- Doing this coding from my Raspi 5 Debian, took a while to get the setup squared away
- General Kaggle setup took a while, abandoned trying to use the CLI API in favor of just using a Kaggle notebook
- Lack of C++ Experience is glaring
- Too many questions and not enough answers on day 1. 

### Performance Observations:
- Code executed instantly (obviously)

### Notes:
- I wonder how many "Hello, Worlds!" that I would have to print in order for execution to not be immediate. My intuition says some large enough multiple of the number of streaming multiprocessors in the GPU itself would create a noticeable time interval between script execution and termination. I am sure this question will be addressed with further and more complex kernels on future days
- Would it be more beneficial to my learning (and my finances) to buy a local GPU to run CUDA on? Which GPU would be most efficient for my purposes?
- Threading syntax in CUDA is so much better than threading syntax in C++
- I need to learn more about how the lock/release system works under the hood (eventually)

## Day 1: Vector Addition Kernel/Naive Matrix Multiplication

### Resources:
- Read Chapter 2.1-2.3 of PMPP
- Claude breakdowns of tough concepts
- Nvidia CUDA documentation
- LeetGPU

### Learnings:
- Re-learned what extern was in C (LeetGPU testcases)
- Learned and practice computed dot products by hand
- Learned the hierarchy of grids, blocks, and threads in CUDA using a childish example
- One big learning for me was the discrening factors between CUDA Kernels and general C code. My instict both today and yesterday was to write a main method to act as an entry point for the CUDA script. Learned that that is NOT how it works today
- Learned basic differences between ``` __global__ ```, ``` __host__```, and ``` __device__ ``` global acts as a decorater for kernel functions. host is used as a decorated for C++ methods that run on the host processor. Device is a decorator for C++ method than can be called from the GPU and run on the kernel (i.e. callable from a thread in the kernel)
- Learned the differnce between ``` cudaMemcpy```, ```cudaMemcpyDeviceToHost```, ```cudaMemcpyHostToDevice```. For some reason, the usage of the term device tripped me up. I now understand that cudaMemcpy is the same as C memcpy but is meant to be used in the GPU kernel. HostToDevice and DeviceToHost act as memcpy from the Host CPU RAM to the Device GPU VRAM.
- Learned cudaMalloc allocates space on the GPU and returns a pointer to an address that only makes sense in the GPU's VRAM
- Breifly read about unified memory and ```cudaMallocManaged(&unified_ptr, size);```

### Challenges
- I'm annoyed with the execution environment of my code
- I am having a hard time inuiting threading

### Performance Observations:
- Both kernels executed instantly which is as expected due to the size of my testcases

### Notes:
- PMPP book makes sense in a macro sense but at my current level there are minimal applications
- Naive matmul kernel seemed TOO simple and I'm certain that I have not written an optimal solution (hence naive)

## Day 2: Lets take a step back - Vector Addition Multiplication Comparison, rethinking my code environment

### Resources:
- Read Chapter 2.4-2.5 of PMPP
- Nvidia CUDA best practices materials

### Learnings:
- Gained much better intuition on how CUDA threads operate by learning from the RGB image color inversion kernel from LeetGPU. Conceptually I understood the way the threads execute but the lack of true iteration in the CUDA code gave me a really hard time to sit down and write CUDA from no examples
- Brushed up on the architecture of a traditional CPU. need to go indepth in GPU architecture and its nuance

### Challenges
- Lack of electrical engineering knowledge makes truly understanding cpu and gpu architecture difficult

### Performance Observations:
- No CUDA executions today as I am working on getting g4dn access

### Notes:
- AWS rules require me to wait for my case to be addressed. I had to request a higher vCPU quota in order to spin up a g4dn.xlarge EC2 instance (I am planning to use Amazon GPUs rather than coding in kaggle. Such a hassle to write and test code in separate files. Can't be bothered)
- Looked into buying a GPU of my own (Was thinking 5070ti or 5080, both feel like they're currently out of my price range given my short term plans, so $0.5/h on AWS is not too bad)
- Got more familiar with C++ code, though I am still very weak

## Day 3: The introduction of AWS g4dn.xlarge instances for real CUDA usage and vector addition simulations

### Resources:
- PMPP 3.1-3.4

### Learnings:
- cudaMalloc(), cudaMemcpy() in depth explanations from PMPP
- Wrote color inversion LeetGPU kernel
- Rewrote naive matmul kernel from memory

### Challenges
- I am reading about these method but have not found good instances in which to use them

### Performance Observations:
- Again, large testcases will be the stress test on my kernels. Still awaiting access to enough vCPUs on AWS to run CUDA there outside of a managed LeetGPU/Kaggle environment

### Notes:
- Made a mistake by only raising my EC2 On-demand G instance quota to 1 vCPU when 4 is the minimum requirement for a g4dn.xlarge instance. Whoops

## Day 4: boom! g4dn liftoff + matrix transpose kernel
Running real CUDA on my EC2 instance

### Resources:
- AWS documentation and GPT for ensuring I'm not going to wake up with a $1000 AWS bill
- PMPP 3.5-3.6

### Learnings:
- More about ```__global__, __device__, __host__```
- Matrix transpose formulas and functions

### Challenges:
- Tried to derive the matrix transpose pattern from scratch
- Still struggling with mapping ```int i = blockDim.x * blockIdx.x + threadIdx.x; 
                                   int j = blockDim.y * blockIdx.y + threadIdx.y;```
    to anything meaningful in my head
    
### Performance Observations:
- I'm amazed at the performance of the vecadd kernel. Instant execution of 100 million elements! I need more memory already to test these limits

### Notes:
- EC2 instance is up and running for CUDA! lets go. 3 days and multiple quota requests later, I have finally run and compiled cuda on a system that I can directly interact with
- Here are the results from the vecAdd CPU vs GPU comparison. The results speak for themselves when it comes to GPU efficiency up to 16gb (My max memory on g4dn.xlarge)
    ```
    ========== Results ==========
    Array size: 10000000 elements
    CPU time: 9 ms
    GPU time: 0 ms
    Speedup: inf
    ========== Results ==========
    Array size: 100000000 elements
    CPU time: 97 ms
    GPU time: 0 ms
    Speedup: inf
    ```
- I am interested to investigate where the extra 7ms came from in CPU execution time (I would assume with 10x elements we should have 10x run time, CPU was closer to 11x)
- Additionally, I am interested to see how different the GPU execution time would be if I were running this code on a 5080 rather than T4s

## Day 5: 2D Matrix addition kernel

### Resources:
- PMPP Chapter 4.1-4.4

### Learnings:
- thread blocks usually are organized into 3 dimensions (x, y, z). Becayse grids are 2D arrays of block dimensions, the third field, Z, should be set to 1 for clarity
- Kernel laugh <<<>>> parameters are, in order, dimensions of grid in terms of number of blocks, and dimensions of each block in terms of number of threads
- Each kernel parameters is a C struct with 3 unsigned integer fields of x, y, z

### Performance Observations:
- When running with this test grid setup (thanks julien), the first kernel launch includes all overhead to launch threads.
```
    // Each test is 1m elements
    Block Size: (1, 1), Time: 1.193824 ms       // has original launch overhead
    Block Size: (32, 32), Time: 0.056608 ms     
    Block Size: (16, 16), Time: 0.055648 ms
    Block Size: (8, 8), Time: 0.055392 ms       // all runs with no launch overhead are similar in speed
  ```


### Notes:
- Learned that more threads is not necessarily good. In the case of 32x32, it is the max legal size. This can limit the processing speed because each block takes up an entire streaming multiprocessor



## Day 6: Reverse Array Kernel

### Resources:
- PMPP Chapter 4.5-4.6

### Learnings:
- Light on the book today

### Performance Observations:
- Instant execution

### Notes:
- First totally independent kernel write, with no resources


## Day 7: Independent Tiled Matrix Multiplication (Device and Host)

### Resources:
- CUDA manual, GPT, W3Schools

### Learnings:
- This was hard. Wrapping my head around the tiling process was super difficult. Implementing the code in CUDA was 10x easier than implementing it in C

### Performance Observations:
```
===== GPU BENCHMARK =====
Block size: (1, 1)
GPU time: 2755.440 ms

Block size: (16, 16)
GPU time: 2.158 ms

Block size: (32, 32)
GPU time: 0.000 ms

===== CPU BENCHMARK =====
CPU time: 122455 ms
```
- Explains itself I think

### Notes:
- Again, 1x1 grid really was just used for initialization
- Got better at using syncthreads after reading about it yesterday




<!--

## Day 5:
_

### Resources:
- 

### Learnings:
- 

### Performance Observations:
- 

### Notes:
- 

-->


To be continued...
