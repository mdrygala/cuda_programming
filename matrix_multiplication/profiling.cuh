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

#include "config.h"
#include "kernels/kernels.cuh"
#include "kernels/baseline.cuh"
#include "kernels/tiling.cuh"
#include "kernels/subtiling_scalar.cuh"
#include "kernels/subtiling_warp_load.cuh"
#include "kernels/subtiling_warp_load_general_dim.cuh"
#include "kernels/tensor_cores.cuh"



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


template <typename InputT>
void launch_kernel(int M, int N, int K,
                   float alpha,
                   const InputT* __restrict__ dA,
                   const InputT* __restrict__ dB,
                   float beta,
                   float* __restrict__ dC,
                   dim3& grid,
                   dim3& block,
                   Config& config){
     static_assert(
        std::is_same<InputT, float>::value ||
        std::is_same<InputT, __half>::value,
        "launch_kernel<InputT>: InputT must be float or __half"
    );
    if (config.kernel_type == "baseline"){
        GEMMBaseline<InputT><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    else if (config.kernel_type == "tiling"){
            GEMMTiling<InputT><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    else if (config.kernel_type == "registerscalar"){
            GEMMSubTilingScalar<InputT><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    else if (config.kernel_type == "warpslab"){
            GEMMSubTilingLoadSlabLinear<InputT><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    else if (config.kernel_type == "warpslabgendim"){
            GEMMSubTilingLoadSlabGenDims<InputT><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    else if constexpr (std::is_same<InputT, float>::value) {
        
        if (config.kernel_type == "registerscalartransposed"){
            GEMMSubTilingScalarTransposed<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        else if (config.kernel_type == "registervec4"){
            GEMMSubTilingVec4<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        else if (config.kernel_type == "registervec4transposed"){
            GEMMSubTilingVec4Transposed<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        else if (config.kernel_type == "warpslabtransposed"){
            GEMMSubTilingLoadSlabLinearTransposed<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        else{
            throw std::runtime_error("Unknown kernel_type");
        }
    }
    else if constexpr(std::is_same<InputT, __half>::value){
        if (config.kernel_type == "tensorcores"){
            GEMMTensorCores<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
    }
    else {
        throw std::runtime_error(
            "Only baseline is implemented for half so far"
        );
    }
}

void set_block_and_grid(dim3& block, dim3& grid, Config& config, int M, int N)
{
    if (config.kernel_type == "baseline" || config.kernel_type == "tiling") {
        block = dim3(TILE, TILE, 1);
        grid  = dim3((N + TILE - 1) / TILE,
                     (M + TILE - 1) / TILE,
                     1);
    }
    else if (config.kernel_type == "registerscalartransposed" ||
             config.kernel_type == "registervec4transposed" ||
             config.kernel_type == "warpslabtransposed"||
             config.kernel_type == "warpslabgendim") {
        block = dim3(NUM_THREADS_X,
                     NUM_THREADS_Y,
                     1);

        grid  = dim3((N + SUBTILE_MN - 1) / SUBTILE_MN,
                     (M + SUBTILE_MN - 1) / SUBTILE_MN,
                     1);
    }
    else if (config.kernel_type == "tensorcores"){
        block = dim3(NUM_THREADS_PER_BLOCK_TENSOR_CORE, 1, 1);
        grid  = dim3((N + SUBTILE_MN - 1) / SUBTILE_MN,
                     (M + SUBTILE_MN - 1) / SUBTILE_MN,
                     1);
    }
    else {
        block = dim3(SUBTILE / SUB,
                     SUBTILE / SUB,
                     1);

        grid  = dim3((N + SUBTILE - 1) / SUBTILE,
                     (M + SUBTILE - 1) / SUBTILE,
                     1);
    }
}






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
    set_block_and_grid(block, grid, config, M, N);

    launch_kernel<InputT>(
        M, N, K,
        alpha,
        dA, dB,
        beta,
        dC,
        grid, block,
        config
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


double peak_compute_flops_per_sm_per_cycle(
    const cudaDeviceProp& prop,
    bool use_tensor_cores)
{
    // A100 = compute capability 8.0
    if (prop.major == 8 && prop.minor == 0) {
        if (use_tensor_cores) {
            return 2048.0;  // A100 FP16 tensor cores with FP32 accumulation, dense peak
        }

        return 128.0;       // normal CUDA-core FP32 FMA roof
    }

    return -1.0;
}


void printRooflineStats(double flops,
                        double theoretical_bytes_transferred,
                        double kernel_time_ms,
                        DataType dtype, bool use_tensor_cores)
{
    const double kernel_time_s = kernel_time_ms / 1000.0;

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    const double peak_bandwidth_gb_s =
        2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1e6;
    const double peak_bandwidth_bytes_s = peak_bandwidth_gb_s * 1e9;

    std::printf("\n--- Roofline Summary ---\n");
    std::printf("Kernel time:       %.4f ms\n", kernel_time_ms);
    // std::printf("Peak bandwidth:    %.2f GB/s\n", peak_bandwidth_gb_s);

    std::printf("Data type:         ");
    if (dtype == DataType::Float32) {
        std::printf("Float32\n");
    } else if (dtype == DataType::Float16) {
        std::printf("Float16\n");
    } else {
        std::printf("Unknown\n");
    }

    // Memory-bound case
    if (flops == 0.0) {
        const double ideal_time_s =
            theoretical_bytes_transferred / peak_bandwidth_bytes_s;
        const double efficiency = ideal_time_s / kernel_time_s;

        std::printf("Theoretical bytes: %.2f\n", theoretical_bytes_transferred);
        std::printf("Ideal time:        %.6f ms\n", ideal_time_s * 1000.0);
        std::printf("Bandwidth efficiency: %.2f%%\n", 100.0 * efficiency);
        std::printf("-------------------------\n");
        return;
    }

    const double flops_per_sm_per_cycle =
        peak_compute_flops_per_sm_per_cycle(prop, use_tensor_cores);

    // if (flops_per_sm_per_cycle < 0.0) {
    //     std::printf("Unsupported compute capability/datatype: %d.%d, ",
    //                 prop.major, prop.minor);

    //     if (dtype == DataType::Float32) {
    //         std::printf("Float32\n");
    //     } else if (dtype == DataType::Float16) {
    //         std::printf("Float16\n");
    //     } else {
    //         std::printf("Unknown\n");
    //     }

    //     std::printf("Arithmetic intensity: %.2f FLOP/byte\n",
    //                 flops / theoretical_bytes_transferred);
    //     std::printf("-------------------------\n");
    //     return;
    // }

    const double sm_clock_hz = prop.clockRate * 1000.0;

    const double peak_compute_flops_s =
        prop.multiProcessorCount * flops_per_sm_per_cycle * sm_clock_hz;

    const double arithmetic_intensity =
        flops / theoretical_bytes_transferred;

    const double peak_memory_flops_s =
        arithmetic_intensity * peak_bandwidth_bytes_s;

    const double peak_flops_s =
        std::min(peak_compute_flops_s, peak_memory_flops_s);

    const double achieved_flops_s = flops / kernel_time_s;

    const double achieved_tflops_s = achieved_flops_s / 1e12;
    const double peak_compute_tflops_s = peak_compute_flops_s / 1e12;
    const double peak_memory_tflops_s = peak_memory_flops_s / 1e12;
    const double peak_roofline_tflops_s = peak_flops_s / 1e12;

    const double achieved_bandwidth_bytes_s =
        theoretical_bytes_transferred / kernel_time_s;
    const double achieved_bandwidth_tb_s =
        achieved_bandwidth_bytes_s / 1e12;
    const double peak_bandwidth_tb_s =
        peak_bandwidth_bytes_s / 1e12;

    const double efficiency = achieved_flops_s / peak_flops_s;

// std::printf("Arithmetic intensity: %.2f FLOP/byte\n", arithmetic_intensity);
// std::printf("Compute roof:         %.2f TFLOP/s\n", peak_compute_tflops_s);
// std::printf("Memory roof:          %.2f TFLOP/s\n", peak_memory_tflops_s);
std::printf("Achieved:             %.2f TFLOP/s\n", achieved_tflops_s);
std::printf("Peak (roofline):      %.2f TFLOP/s\n", peak_roofline_tflops_s);
std::printf("Efficiency:           %.2f%%\n", 100.0 * efficiency);
std::printf("-------------------------\n");
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
    set_block_and_grid(block, grid, config, M, N);

    constexpr int NUM_WARMUPS = 40;
    constexpr int NUM_REPEATS = 10;

    for (int i = 0; i < NUM_WARMUPS; ++i) {
        launch_kernel<InputT>(
            M, N, K,
            alpha,
            dA, dB,
            beta,
            dC,
            grid, block,
            config
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
            grid, block,
            config
        );
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaProfilerStop());

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    float milliseconds = total_ms / NUM_REPEATS;

    double flops = 2.0 * M * N * K;
    double seconds = milliseconds / 1000.0;
    double gflops = (flops * 1e-9) / seconds;
    double tflops = gflops / 1000.0;

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


    bool use_tensor_cores =
    config.kernel_type.find("tensor") != std::string::npos;

    printRooflineStats(
        flops,
        theoretical_bytes_transferred,
        milliseconds,
        config.data_type,
        use_tensor_cores
    );

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    printf("Profiling loop finished.\n");
}