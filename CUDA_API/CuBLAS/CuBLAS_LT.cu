#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <functional>
#include <iomanip>
#include <random>
#include <numeric>

#define CHECK_CUDA(call){  \
    cudaError_t status = call; \
    if(status != cudaSuccess){ \
        std::cerr <<"Cuda error at line "<< __LINE__<< ":" << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call){ \
    cublasStatus_t status = call; \
    if (status == CUBLAS_STATUS_SUCCESS){ \
        std::cerr <<"cuBLAS error at line: " << __LINE__ << ":" << status << std :: endl; \
    } \
}

const int M = 4096;
const int K = 1024;
const int N = 4096;

// Naive CUDA kernel for Matrix Multiplication
__global__ void naiveMatrixMultiply(float* A, float* B, float* C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < M && col < N){
        float sum = 0.0f;
        for (int i = 0; i < K; ++i){
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// function to initialize matrix with random values
void initializeMatrix(std::vector<float>& matrix, int rows, int cols){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    for (int i = 0; i < rows * cols; ++i){
        matrix[i] = static_cast<float>(dis(gen));
    }
}


bool verifyResults(const std::vector<float>& expected, const std::vector<float>& actual, float tolerance = 1e-2){
    if(expected.size() != actual.size()){
        return false;
    }
    for(size_t i = 0; i < expected.size(); ++i){
        float rel_error = std::abs(expected[i] - actual[i]);
        if (rel_error > tolerance){
            std::cout << "Mismatch at index " << i << ": expected "<< expected[i]
                      <<", got " << actual[i] << ", relative error: " << rel_error << std::endl;
            return false;
        }
    }
    return true;
}

//New function for CUDA event-based timing
float time_kernel(std::function<void()> kernel_func){
    cudaEvent_t start, stop;
    float elapsed_time;
    
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel_func();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop))

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));


    return elapsed_time;
}

float benchmark_kernel(std::function<void()> kernel_func, int warmup_runs, int benchmark_runs){
    // warmup_runs
    for(int i = 0; i< warmup_runs; ++i){
        kernel_func();
    }

    // Benchmark runs
    std::vector<float> times;
    for (int i = 0; i < benchmark_runs; ++i){
        float time = time_kernel(kernel_func);
        times.push_back(time);
    }

    //calculate average time
    float avg_time = std::accumulate(times.begin(), times.end(), 0.0f)/ benchmark_runs;
    return avg_time;
}   

int main(){
    std::cout <<"matrix size: " << M << "x" << K << "x" << N << std::endl;
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    std::vector<float> h_C_cublas_fp32(M * N), h_C_cublasLT_fp32(M * N);
    std::vector<float> h_C_cublas_fp16(M * N), h_C_cublasLT_fp16(M * N);
    std::vector<float> h_C_naive(M * N);
    

    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);

    float *d_A, *d_B, *d_C;
    half *d_A_half, *d_B_half, *d_C_half;

    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_A_half, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_half, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_half, M * N * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));

    //Convert to half
    std::vector<half> h_A_half(M * K), h_B_half(K * N);
    for(int i = 0; i < M * K; ++i)h_A_half[i] = __float2half(h_A[i]);
    for(int i = 0; i < K * N; ++i)h_B_half[i] = __float2half(h_B[i]);

    CHECK_CUDA(cudaMemcpy(d_A_half, h_A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_half, h_A_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    cublasLtHandle_t cublasLt_handle;
    CHECK_CUBLAS(cublasLtCreate(&cublasLt_handle));
    
    float alpha = 1.0f , beta = 0.0f;\
    half alpha_half = __float2half(alpha);
    half beta_half = __float2half(beta);

    const int warmup_runs = 3;
    const int benchmark_runs = 20;

    //cuBLAS FP32
    float cublas_fp32_time = benchmark_kernel([&](){
        CHECK_CUBLAS(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
        }, warmup_runs, benchmark_runs);
    std::cout << "cuBLAS FP32 average time: " << cublas_fp32_time << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_C_cublas_fp32.data(), d_C, M * N *sizeof(float), cudaMemcpyDeviceToHost));


    // cuBLASLt FP32
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc,CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, M));

    float cublasLt_fp16_time = benchmark_kernel([&]() {
        CHECK_CUBLAS(cublasLtMatmul(cublasLt_handle, operationDesc, &alpha, d_B, Bdesc, d_A, Adesc, &beta, d_C, Cdesc, d_C, Cdesc, nullptr, nullptr, 0, 0));
    }, warmup_runs, benchmark_runs);
    std::cout << "cuBLASLt FP16 average time: " << cublasLt_fp16_time << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_C_cublas_fp32.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // cuBLAS fp16
    float cublas_fp16_time = benchmark_kernel([&](){
        CHECK_CUBLAS(cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_half, d_B_half, N, d_A_half, K, &beta_half, d_C_half, N));
    }, warmup_runs, benchmark_runs);
    std::cout << "cuBLAS FP16 average time: " << cublas_fp16_time << " ms" << std::endl;

    std::vector<half> h_C_half(M * N);
    CHECK_CUDA(cudaMemcpy(h_C_half.data(), d_C_half,M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for(int i = 0; i < M * N; ++i) h_C_cublas_fp16[i] = __half2float(h_C_half[i]);

    //cuBLASLt FP16
    cublasLtMatmulDesc_t operationDesc_half = nullptr;
    cublasLtMatrixLayout_t Adesc_half = nullptr, Bdesc_half = nullptr, Cdesc_half = nullptr;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&operationDesc_half, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc_half, CUDA_R_16F, K, M, K));     
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc_half, CUDA_R_16F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc_half, CUDA_R_16F, M, N, M));

    float cublasLt_fp16_time = benchmark_kernel([&](){
        CHECK_CUBLAS(cublasLtMatmul(cublasLt_handle, operationDesc_half, &alpha_half, d_B_half, Bdesc_half, d_A_half, Adesc_half, &beta_half, d_C_half, Cdesc_half, d_C_half, Cdesc_half, nullptr, nullptr, 0, 0));
        }, warmup_runs, benchmark_runs);
        std::cout << "cuBLASTLt FP16 average time: " << cublasLt_fp16_time << " ms" <<std::endl;
    
    CHECK_CUDA(cudaMemcpy(h_C_half.data(), d_C_half, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for(int i= 0; i <M* N; ++i) h_C_cublas_fp16[i] =  __half2float(h_C_half[i]);

    // Naive CUDA kernel
    dim3 blockDim(32,32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    float naive_cuda_time = benchmark_kernel([&](){
        naiveMatrixMultiply<<< gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        }, warmup_runs, benchmark_runs);
    std::cout << "Naive CUDA kernel average time: " << naive_cuda_time << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_C_naive.data(), d_C, M * n * sizeof(float), cudaMemcpyDeviceToHost));

    //Verify results
    bool cublas_fp32_correct = verifyResults(h_C_naive, h_C_cublas_fp32, 1e-2);

    bool cublasLt_fp32_correct = verifyResults(h_C_naive, h_C_cublasLT_fp32, 1e-2);

    bool cublas_fp16_correct = verifyResults(h_C_naive, h_C_cublas_fp16, 1e-2);
    
    bool cublasLt_fp16_correct = verifyResults(h_C_naive, h_C_cublasLt_fp16, 1e-2);

    
    //compute the max error for each of the fp16 results
    float max_error_fp16_cublas = 0.0f;
    float max_error_fp16_cublasLt = 0.0f;

    for(int i = 0; i < M * N; ++i){
        float error = std::abs(h_C_naive[i] - h_C_cublas_fp16[i]);
        if (error > max_error_fp16_cublas){
            max_error_fp16_cublas = error;
        }
    }

    for(int i = 0; i < M * N; ++i){
        float error = std::abs(h_C_naive[i] - h_C_cublasLt_fp16[i]);
        if (error > max_error_fp16_cublasLt){
            max_error_fp16_cublasLt = error;
        }
    }

    std::cout<<"max error fp16 cublas: " << max_error_fp16_cublas << std::endl;
    std::cout<<"max error fp16 cublasLt: " << max_error_fp16_cublasLt << std::endl;


    
    std::cout << "cuBLAS FP32 results " << (cublas_fp32_correct ? "match" : "do not match") << " the naive kernel results within tolerance of 1e-2." << std::endl;
    std::cout << "cuBLASLt FP32 results " << (cublasLt_fp32_correct ? "match" : "do not match") << " the naive kernel results within tolerance of 1e-2." << std::endl;
    std::cout << "cuBLAS FP16 results " << (cublas_fp16_correct ? "match" : "do not match") << " the naive kernel results within tolerance of 5e-1." << std::endl;
    std::cout << "cuBLASLt FP16 results " << (cublasLt_fp16_correct ? "match" : "do not match") << " the naive kernel results within tolerance of 5e-1." << std::endl;

        // Clean up
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(operationDesc_half));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Adesc_half));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Bdesc_half));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(Cdesc_half));
    CHECK_CUBLAS(cublasLtDestroy(cublasLt_handle));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_half));
    CHECK_CUDA(cudaFree(d_B_half));
    CHECK_CUDA(cudaFree(d_C_half));

    
}
