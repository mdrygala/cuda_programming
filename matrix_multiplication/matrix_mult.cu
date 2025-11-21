//Multplies two matricies  on the GPU using CUDA.

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include "kernels.cuh"
#include "../include/benchmarking.hpp"
#include <cublas_v2.h>

// Fill matrices A (M×K) and B (K×N) with random ints in [0,100),
// and C (M×N) with zeros
void initializeMatrices(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M*K; i++) {
        A[i] = static_cast<float>(rand() % 100);
    }
    for (int i = 0; i < K*N; i++) {
        B[i] = static_cast<float>(rand() % 100);
    }
    for (int i = 0; i < M*N; i++) {
        C[i] = static_cast<float>(rand() % 100); // nonzero init
    }
}

// Verify GEMM result on CPU: check C ≈ alpha*A*B + beta*Cinit
void verifyGEMM(const float *A, const float *B,
                const float *Cgpu, const float *Cinit,
                int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float ref = 0.0f;
            for (int k = 0; k < K; k++) {
                ref += A[i*K + k] * B[k*N + j];
            }
            ref = alpha * ref + beta * Cinit[i*N + j];
            assert(fabs(Cgpu[i*N + j] - ref) < 1e-3f);
        }
    }
}

// void resetC(int* C, const int* Cinit, size_t M, size_t N) {
//     size_t bytes = M * N * sizeof(float);
//     cudaMemcpy(C, Cinit, bytes, cudaMemcpyDeviceToDevice);
// }


static inline cublasHandle_t make_cublas(cudaStream_t st=0){
    cublasHandle_t h; cublasCreate(&h); cublasSetStream(h, st);
    // For pure FP32 on CUDA cores (fair FP32 compare):
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    // If you want TF32 tensor cores instead, flip this one line:
    // cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    return h;
}
// Row-major wrapper: C = A*B  (maps to cuBLAS column-major call)
static inline void sgemm_rowmajor(cublasHandle_t h,
                                  int M,int N,int K,
                                  float alpha, const float* A,const float* B,
                                  float beta,        float* C)
{
    // Treat row-major as column-major by swapping A/B and M/N
    // C^T = B^T * A^T
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                /*m=*/N, /*n=*/M, /*k=*/K,
                &alpha,
                /*A=*/B, /*lda=*/N,
                /*B=*/A, /*ldb=*/K,
                &beta,
                /*C=*/C, /*ldc=*/N);
}


int main(){
    if ((SUBTILE % SUB) != 0) {
    fprintf(stderr, "Error: TILE (%d) must be divisible by SUB (%d)\n", TILE, SUB);
    return EXIT_FAILURE;
    }

    // Print device informationn
    int sms = readSMs();
    double peakGBs = computePeakGBs();
    printf("peak GBS %.3f\n", peakGBs);
    int RegSM = readRegSm();
    printf("SMs %d and Registers per SM %d\n", sms, RegSM);
    print_sm_caps();
    printf("Theoretical FP32 peak: %.2f GFLOP/s\n", peakFP32_GFLOPs());

    // Set our problem size
    int N = 1<<13;
    int M = N;
    int K = N<<1;
    float alpha = 1.0f;
    float beta = 1.0f;
    size_t ABytes = M * K * sizeof(float);
    size_t BBytes = K * N * sizeof(float);
    size_t CBytes = M * N * sizeof(float);
    size_t totalBytes = ABytes + BBytes + 2.0 * CBytes;

    //Allocate unified memory for matricies
    float *A, *B, *C, *Cinit;

    cudaMallocManaged(&A, ABytes);
    cudaMallocManaged(&B, BBytes);
    cudaMallocManaged(&C, CBytes);
    cudaMallocManaged(&Cinit, CBytes);

    // Initialize input vectors
    initializeMatrices(A, B, Cinit, M, N, K);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE- 1) / TILE, (M + TILE- 1) / TILE);

    double gflops_total = 2.0 * M * N * K / 1e9;
    
    // run Baseline kernel
    
    auto launch_reset_baseline = [&](){
        cudaMemcpyAsync(C, Cinit, CBytes, cudaMemcpyDeviceToDevice);
        GEMMBaseline<<<grid, block>>>(M,N,K, alpha, A, B, beta, C);
    };

    
    float ms = bench_best_ms(launch_reset_baseline);
    printf("Kernel time for Baseline : %.3f ms  |  %.2f GFLOP/s\n", ms, gflops_total / (ms / 1e3));

    // run tiling kernel
  

    auto launch_reset_tiling = [&](){
        cudaMemcpyAsync(C, Cinit, CBytes, cudaMemcpyDeviceToDevice);
        GEMMTiling<<<grid, block>>>(M,N,K, alpha, A, B, beta, C);
    };

    
    ms = bench_best_ms(launch_reset_tiling);
    printf("Kernel time for Basic Tiling : %.3f ms  |  %.2f GFLOP/s\n", ms, gflops_total / (ms / 1e3));

    // run subtile kernel
    dim3 blockSub(SUBTILE/SUB, SUBTILE/SUB);
    dim3 gridSub((N + SUBTILE- 1) / SUBTILE, (M + SUBTILE- 1) / SUBTILE);

    auto launch_reset_subtiling_naive = [&](){
        cudaMemcpyAsync(C, Cinit, CBytes, cudaMemcpyDeviceToDevice);
        GEMMSubTiling<LoaderNaive, ComputeNaive><<<gridSub, blockSub>>>(M,N,K, alpha, A, B, beta, C);
    };

    
    ms = bench_best_ms(launch_reset_subtiling_naive);
    printf("Kernel time for Register Sub Tiling (Naive) : %.3f ms  |  %.2f GFLOP/s\n", ms, gflops_total / (ms / 1e3));



    auto launch_reset_subtiling_vec4 = [&](){
        cudaMemcpyAsync(C, Cinit, CBytes, cudaMemcpyDeviceToDevice);
        GEMMSubTiling<LoaderVec4, ComputeNaive><<<gridSub, blockSub>>>(M,N,K, alpha, A, B, beta, C);
    };

    
    ms = bench_best_ms(launch_reset_subtiling_vec4);
    printf("Kernel time for Register Sub Tiling (Vectorized Loading) : %.3f ms  |  %.2f GFLOP/s\n", ms, gflops_total / (ms / 1e3));


    auto launch_reset_subtiling_swizzle = [&](){
        cudaMemcpyAsync(C, Cinit, CBytes, cudaMemcpyDeviceToDevice);
        GEMMSubTiling<LoaderVec4Swizzle, ComputeSwizzle><<<gridSub, blockSub>>>(M,N,K, alpha, A, B, beta, C);
    };

    
    ms = bench_best_ms(launch_reset_subtiling_swizzle);
    printf("Kernel time for Register Sub Tiling (Vectorized Loading Swizzle) : %.3f ms  |  %.2f GFLOP/s\n", ms, gflops_total / (ms / 1e3));


    cublasHandle_t h = make_cublas();

    
    auto launch_reset_cublas = [&](){
        cudaMemcpyAsync(C, Cinit, CBytes, cudaMemcpyDeviceToDevice);
        sgemm_rowmajor(h, M, N, K, alpha, A, B, beta, C);
    };

    //
    ms = bench_best_ms(launch_reset_cublas, /*warmups=*/3, /*iters=*/50);
    printf("cuBLAS (FP32 cores) : %.3f ms  |  %.2f GFLOP/s\n",
        ms, gflops_total / (ms / 1e3));

    //
    cublasDestroy(h);
    
   

    //Free GPU Memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);


    return 0;
}
