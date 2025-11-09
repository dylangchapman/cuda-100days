# 100 Days of Writing CUDA Kernels

- Inspired by julienokumu
- Cannot run CUDA on my machine, so will be using Kaggle/Colab free GPUs
- This will be a long process of getting more familiar with C++, CUDA, and Linear Algebra

## Day 0:
Hello GPU World Kernel

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

### Thoughts:
- I wonder how many "Hello, Worlds!" that I would have to print in order for execution to not be immediate. My intuition says some large enough multiple of the number of streaming multiprocessors in the GPU itself would create a noticeable time interval between script execution and termination. I am sure this question will be addressed with further and more complex kernels on future days
- Would it be more beneficial to my learning (and my finances) to buy a local GPU to run CUDA on? Which GPU would be most efficient for my purposes?
- Threading syntax in CUDA is so much better than threading syntax in C++
- I need to learn more about how the lock/release system works under the hood

## Day 1:
Vector Addition Kernel

### Resources:
Read Chapter 2.1-2.3 of PMPP

To be continued...
