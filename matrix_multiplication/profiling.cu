#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <iostream>

#include "config.h"
#include "kernels.cuh"




#define CHECK_CUDA(call) do {                                   \
  cudaError_t err = (call);                                     \
  if (err != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
            __FILE__, __LINE__, cudaGetErrorString(err));        \
    std::exit(1);                                               \
  }                                                             \
} while(0)

static void verifyGEMM_small(const float *A, const float *B,
                             const float *Cgpu, const float *Cinit,
                             int M, int N, int K, float alpha, float beta)
{
    const float atol = 1e-2f, rtol = 1e-2f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float ref = 0.0f;
            for (int k = 0; k < K; k++) ref += A[i*K + k] * B[k*N + j];
            ref = alpha * ref + beta * Cinit[i*N + j];

            float diff = std::fabs(Cgpu[i*N + j] - ref);
            float tol  = atol + rtol * std::fabs(ref);
            if (diff > tol) {
                fprintf(stderr,
                        "Mismatch (%d,%d): gpu=%f ref=%f diff=%f tol=%f\n",
                        i, j, Cgpu[i*N+j], ref, diff, tol);
                std::exit(2);
            }
        }
    }
}

void parseArgs(int argc, char** argv, Config& config);
void set_block_and_grid(dim3& block, dim3& grid, Config& config, int M, int N);
void launch_kernel(int M,int N,int K,
                                 float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C,  dim3& grid, dim3& block, Config& config);

int main(int argc, char** argv) {
    static_assert(SUBTILE % (SUB * SUB) == 0, "SUB^2 must divide SUBTILE");
    Config config;
    parseArgs(argc, argv, config);
    
    

    // ---------- basic config sanity ----------
    assert(SUBTILE % SUB == 0);

    // ---------- 1) VERIFY ON SMALL ----------
    {
        int M = 256, N = 256, K = 256;
        float alpha = 1.0f, beta = 1.0f;

        std::vector<float> A(M*K, 1.0f);
    
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                A[i*K + k] = float(k + 1);
            }
        }
        std::vector<float> B(K*N, 1.0f);
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                B[k*N + j] = float(k + 1);
            }
        }
        std::vector<float> C(M*N, 0.0f);
        std::vector<float> Cinit = C;

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CHECK_CUDA(cudaMalloc(&dA, sizeof(float)*M*K));
        CHECK_CUDA(cudaMalloc(&dB, sizeof(float)*K*N));
        CHECK_CUDA(cudaMalloc(&dC, sizeof(float)*M*N));

        CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dC, C.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));

        dim3 block, grid;

        set_block_and_grid(block, grid, config, M, N);

        launch_kernel(M, N, K, alpha, dA, dB, beta, dC, grid, block, config);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(C.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
        verifyGEMM_small(A.data(), B.data(), C.data(), Cinit.data(), M, N, K, alpha, beta);

        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));


        printf("Verification passed.\n");
    }
   
    // ---------- 2) PROFILE ON LARGE (ONLY THIS REGION) ----------
    {
        int N = 1 << 11;
        int M = N, K = N;
        float alpha = 1.0f, beta = 1.0f;

        std::vector<float> A(M*K, 1.0f);
        std::vector<float> B(K*N, 1.0f);
        std::vector<float> C(M*N, 0.0f);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CHECK_CUDA(cudaMalloc(&dA, sizeof(float)*M*K));
        CHECK_CUDA(cudaMalloc(&dB, sizeof(float)*K*N));
        CHECK_CUDA(cudaMalloc(&dC, sizeof(float)*M*N));

        CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dC, C.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));

        dim3 block, grid;
        set_block_and_grid(block, grid, config, M, N);

        // 1. Warmup
        launch_kernel(M, N, K, alpha, dA, dB, beta, dC, grid, block, config);
        // Kernel<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 2. Instrument with CUDA Events
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaProfilerStart());
        CHECK_CUDA(cudaEventRecord(start));
        
        launch_kernel(M, N, K, alpha, dA, dB, beta, dC, grid, block, config);
        
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaProfilerStop());

        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

        // 3. GOODPUT MATH
        // Formula: (2 * M * N * K) / (Time in seconds)
        double flops = 2.0 * M * N * K;
        double seconds = milliseconds / 1000.0;
        double gflops = (flops * 1e-9) / seconds;
        double tflops = gflops / 1000.0;

        // Theoretical Peak for A100-SXM4-40GB (FP32) is ~19.5 TFLOPS
        double a100_peak = 19.5; 
        double efficiency = (tflops / a100_peak) * 100.0;

        printf("\n--- Goodput Statistics ---\n");
        printf("Problem Size: %d x %d x %d\n", M, N, K);
        printf("Kernel Time:  %.4f ms\n", milliseconds);
        printf("Achieved:     %.2f GFLOPS (%.2f TFLOPS)\n", gflops, tflops);
        printf("Efficiency:   %.2f%% of Theoretical Peak (19.5 TFLOPS)\n", efficiency);
        printf("--------------------------\n");

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));

        printf("Profiling loop finished.\n");
    }

    return 0;
}


void parseArgs(int argc, char** argv, Config& config){
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--kernel" && i + 1 < argc){
            config.kernel_type = argv[++i];

        }
}
}

void set_block_and_grid(dim3& block, dim3& grid, Config& config, int M, int N){
    if (config.kernel_type == "baseline" || config.kernel_type == "tiling") {
            block = dim3(TILE, TILE, 1);
            grid  = dim3((N + TILE - 1)/TILE, (M + TILE - 1)/TILE, 1);     
        } else {
            block = dim3(SUBTILE / SUB, SUBTILE / SUB, 1);
            grid  = dim3((N + SUBTILE - 1)/SUBTILE, (M + SUBTILE - 1)/SUBTILE, 1);
        }
}

void launch_kernel(int M,int N,int K,
                                 float alpha,
                                 const float* __restrict__ dA,
                                 const float* __restrict__ dB,
                                 float beta,
                                 float* __restrict__ dC, dim3& grid, dim3& block, Config& config)
{
    if (config.kernel_type == "baseline"){
        GEMMBaseline<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    else if (config.kernel_type == "tiling"){
        GEMMTiling<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    else if (config.kernel_type == "registernaive"){
        GEMMSubtileRegNaive<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    else if (config.kernel_type == "registervec4"){
        GEMMSubtileRegVec4<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    else if (config.kernel_type == "warpslab"){
        GEMMSubtileWarpSlab<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    else if (config.kernel_type == "swizzle"){
        GEMMSubtileSwzl<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    else{
        throw std::runtime_error("Unknown kernel_type");
    }
}