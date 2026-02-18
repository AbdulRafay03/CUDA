// kernel profiling: helps to measure how well the kernel runs
    // You write a kernel → it runs → but you don’t know:
        // Is it fast or slow?
        // Is GPU actually busy or mostly idle?
        // Where is time being wasted?
    // Profiling answers these questions.

// we use NVTX to label the kernel for profiling
// download Nvidia Nsight to analyze code
// nsys profile --stats=true filename

// same matmul code but with nvtx

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

#define BLOCK_SIZE 16
 
__global__ void matrixMulKernel(float* A, float* B, float* C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < N && col < N){
        for (int i = 0; i < N; i++){
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMul(float* A, float* B, float* C, int N){
    nvtxRangePush("Matrix Multiplication");

    float *d_A, *d_B, *d_c;
    int size = N * N * sizeof(float);

    nvtxRangePush("Memory Allocation");
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    nvtxRangePop();

    nvtxRangePush("Memory Copy H2D");
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, A, size, cudaMemcpyHostToDevice);
    nvtxRangePop();

    dim3 threadPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    nvtxRangePush("Kernel Execution");
    matrixMulKernel<<<numBlocks, threadPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    nvtxRangePush("Memory Copy D2H");
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToDevice);
    nvtxRangePop();

    nvtxRangePush("Memory Deallocation");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();

    nvtxRangePop();  // End of Matrix Multiplication

}