#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream.h>

#define N 10000000
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

void vector_add_cpu(float *a, float *b, float *c, int n){
    for (int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}


__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i < nx && j < ny && k < nz){
        int idx= i + j * nx + k * nx * ny;
        if (idx < nx * ny * nz){
            c[idx] = a[idx] + b[idx];
        }
    }
}

void init_vector(float *vec, int n){
    for (int i = 0; i < n; i++){
        vec[i] = (float)rand() / RAND_MAX;
    }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC< &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    float *h_a, *h_a, *h_c_cpu, h_c_gpu1d, h_c_gpu_3d;
    float *d_a, *d_b, *d_c_1d, *d_c_3d;
    size_t size = N *sizeof(float);
    
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu_1d = (float*)malloc(size);
    h_c_gpu_3d = (float*)malloc(size);


    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_a, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_1d, size);
    cudaMalloc(&d_c_3d, size);


    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);


    int num_blocks_1d = (N+BLOCK_SIZE_1D-1) / BLOCK_SIZE_1D;

    int nx = 100, ny = 100, nz = 1000;
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d,
        (ny + block_size_3d.y - 1) / block_size_3d,
        (nz + block_size_3d.z - 1) / block_size_3d,
    )

    printf("perform Warmup");
    for (int i = 0; i< 3; i++){
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu_1d<<<num_block_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        vector_add_gpu_3d<<<num_block_3d, BLOCK_SIZE_3D>>>(d_a, d_b, d_c_3d, nx, ny,nz);
        cudaDeviceSynchronize();
    }

    

}