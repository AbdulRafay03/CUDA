// When a process cant be optimized by parallel computing
// Need to complete previous tasks before accessing memory location

// 1- locked address
// 2- Read value
// 3- compute
// 4- unlock
// 5- update

#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_THREADS 1000
#define NUM_BLOCKS 1000

// Kernel without atomics (incorrect)

__global__ void incrementCounterNonAtomic(int* counter){
    //not locked 
    int old = *counter;
    int new_value = old + 1;
    //not unlocked
    *counter = new_value;
}

// Kernel with atomics(correct)

__global__ void incrementCounterAtomic(int* counter){
    int a = atomicAdd(counter, 1);
}

int main(){
    int h_counterNonAtomic = 0;
    int h_counterAtomic = 0;
    int *d_counterNonAtomic, *d_counterAtomic;

    // Allocate device memory
    cudaMalloc((void**)&d_counterNonAtomic, sizeof(int));
    cudaMalloc((void**)&d_counterAtomic, sizeof(int));

    // copy initial counter values to device
    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int),cudaMemcpyHostToDevice);

    // Launch Kernels
    incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
    incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);

    //Copy resutls back to host
    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);

    //print results
    printf("Non-atomic counter value: %d\n", h_counterNonAtomic);
    printf("Atomic counter value: %d\n", h_counterAtomic);

    //Free device memory
    cudaFree(d_counterNonAtomic);
    cudaFree(d_counterAtomic);

    // Non-atomic counter value: 49
    // Atomic counter value: 1000000

}