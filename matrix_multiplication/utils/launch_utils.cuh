#pragma once

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <type_traits>

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "utils/roofline_utils.cuh"

#define CHECK_CUDA(call) do {                                   \
  cudaError_t err = (call);                                     \
  if (err != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
            __FILE__, __LINE__, cudaGetErrorString(err));        \
    std::exit(1);                                               \
  }                                                             \
} while(0)



template <typename T>
static inline float to_float_host(T x)
{
    return static_cast<float>(x);
}

template <>
inline float to_float_host<__half>(__half x)
{
    return __half2float(x);
}

template <typename T>
static inline T from_float_host(float x)
{
    return static_cast<T>(x);
}

template <>
inline __half from_float_host<__half>(float x)
{
    return __float2half(x);
}


template <typename InputT>
static void verifyGEMM_small(const InputT *A, const InputT *B,
                             const float *Cgpu, const float *Cinit,
                             int M, int N, int K, float alpha, float beta)
{
    const float atol = 1e-2f, rtol = 1e-2f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float ref = 0.0f;
            for (int k = 0; k < K; k++){
                float a = to_float_host<InputT>(A[i * K + k]);
                float b = to_float_host<InputT>(B[k * N + j]);

                ref = fmaf(a, b, ref);
            }
            
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




template <int TILE_SIZE>
inline void set_block_and_grid_basic(dim3& block, dim3& grid, int M, int N){
    block = dim3(TILE_SIZE, TILE_SIZE, 1);
    grid  = dim3((N + TILE_SIZE - 1) / TILE_SIZE,
                     (M + TILE_SIZE - 1) / TILE_SIZE,
                     1);

}

template <int TILE_SIZE_M, int TILE_SIZE_N, int NUM_THREADS_N, int NUM_THREADS_M>
inline void set_block_and_grid_subtile(dim3& block, dim3& grid, int M, int N){
    block = dim3(NUM_THREADS_N,
                     NUM_THREADS_M,
                     1);

    grid  = dim3((N + TILE_SIZE_N - 1) / TILE_SIZE_N,
                     (M + TILE_SIZE_M - 1) / TILE_SIZE_M,
                     1);

}

template <int TILE_SIZE_M, int TILE_SIZE_N, int NUM_THREADS>
inline void set_block_and_grid_linear_block(dim3& block, dim3& grid, int M, int N){
    block = dim3(NUM_THREADS,
                     1,
                     1);

    grid  = dim3((N + TILE_SIZE_N - 1) / TILE_SIZE_N,
                     (M + TILE_SIZE_M - 1) / TILE_SIZE_M,
                     1);

}


void parseArgs(int argc, char** argv, Config& config){
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--datatype"  && i + 1 < argc){
            std::string datatype = argv[++i];

            if (datatype == "float" || datatype == "fp32") {
                config.data_type = DataType::Float32;
            }
            else if (datatype == "half" || datatype == "fp16") {
                config.data_type = DataType::Float16;
            }
            else {
                throw std::runtime_error("Unknown --datatype. Use float/fp32 or half/fp16.");
            }
        }
}
}


void set_block_and_grid(dim3& block, dim3& grid, int M, int N);

template <typename InputT>
void launch_kernel(int M, int N, int K,
                   float alpha,
                   const InputT* __restrict__ dA,
                   const InputT* __restrict__ dB,
                   float beta,
                   float* __restrict__ dC,
                   dim3& grid,
                   dim3& block);

template <typename InputT>
void run_verification(int M, int N, int K, Config& config)
{
    float alpha = 1.0f;
    float beta  = 1.0f;

    std::vector<InputT> A(M * K);

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            A[i * K + k] = from_float_host<InputT>(float(k + 1));
        }
    }

    std::vector<InputT> B(K * N);

    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            B[k * N + j] = from_float_host<InputT>(float(k + 1));
        }
    }

    std::vector<float> C(M * N, 0.0f);
    std::vector<float> Cinit = C;

    InputT *dA = nullptr;
    InputT *dB = nullptr;
    float  *dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeof(InputT) * M * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(InputT) * K * N));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float)  * M * N));

    CHECK_CUDA(cudaMemcpy(dA, A.data(),
                          sizeof(InputT) * M * K,
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dB, B.data(),
                          sizeof(InputT) * K * N,
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dC, C.data(),
                          sizeof(float) * M * N,
                          cudaMemcpyHostToDevice));

    dim3 block, grid;
    set_block_and_grid(block, grid, M, N);

    launch_kernel<InputT>(
        M, N, K,
        alpha,
        dA, dB,
        beta,
        dC,
        grid, block
    );

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(C.data(), dC,
                          sizeof(float) * M * N,
                          cudaMemcpyDeviceToHost));

    verifyGEMM_small<InputT>(
        A.data(), B.data(),
        C.data(), Cinit.data(),
        M, N, K,
        alpha, beta
    );

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    printf("Verification passed.\n");
}






template <typename InputT>
void run_profile(int M, int N, int K, Config& config)
{
    float alpha = 1.0f;
    float beta  = 0.0f;

    std::vector<InputT> A(M * K);
    std::vector<InputT> B(K * N);
    std::vector<float> C(M * N, 0.0f);

    for (int i = 0; i < M * K; i++) {
        A[i] = from_float_host<InputT>(1.0f);
    }

    for (int i = 0; i < K * N; i++) {
        B[i] = from_float_host<InputT>(1.0f);
    }

    InputT *dA = nullptr;
    InputT *dB = nullptr;
    float  *dC = nullptr;

    CHECK_CUDA(cudaMalloc(&dA, sizeof(InputT) * M * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(InputT) * K * N));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float)  * M * N));

    CHECK_CUDA(cudaMemcpy(dA, A.data(),
                          sizeof(InputT) * M * K,
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dB, B.data(),
                          sizeof(InputT) * K * N,
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dC, C.data(),
                          sizeof(float) * M * N,
                          cudaMemcpyHostToDevice));

    dim3 block, grid;
    set_block_and_grid(block, grid, M, N);

    constexpr int NUM_WARMUPS = 40;
    constexpr int NUM_REPEATS = 10;

    for (int i = 0; i < NUM_WARMUPS; ++i) {
        launch_kernel<InputT>(
            M, N, K,
            alpha,
            dA, dB,
            beta,
            dC,
            grid, block
        );
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaProfilerStart());
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < NUM_REPEATS; ++i) {
        launch_kernel<InputT>(
            M, N, K,
            alpha,
            dA, dB,
            beta,
            dC,
            grid, block
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaProfilerStop());

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    float milliseconds = total_ms / NUM_REPEATS;

    double flops = 2.0 * M * N * K;

    // Theoretical global-memory traffic for one GEMM:
    // read A + read B + write C.
    // If beta != 0, C is also read.
    double theoretical_bytes_transferred =
        double(M) * double(K) * sizeof(InputT) +
        double(K) * double(N) * sizeof(InputT) +
        double(M) * double(N) * sizeof(float);

    if (beta != 0.0f) {
        theoretical_bytes_transferred += double(M) * double(N) * sizeof(float);
    }




    printRooflineStats(
    flops,
    theoretical_bytes_transferred,
    milliseconds,
    config
);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    printf("Profiling loop finished.\n");
}

inline void run_selected_datatype(Config& config,
                                  int verifyM = 256,
                                  int verifyN = 256,
                                  int verifyK = 256,
                                  int profileM = 1 << 12,
                                  int profileN = 1 << 12,
                                  int profileK = 1 << 12)
{
    if (config.data_type == DataType::Float32) {
        run_verification<float>(verifyM, verifyN, verifyK, config);
        run_profile<float>(profileM, profileN, profileK, config);
    }
    else if (config.data_type == DataType::Float16) {
        run_verification<__half>(verifyM, verifyN, verifyK, config);
        run_profile<__half>(profileM, profileN, profileK, config);
    }
    else {
        throw std::runtime_error("Only allow datatype of float or half");
    }
}

