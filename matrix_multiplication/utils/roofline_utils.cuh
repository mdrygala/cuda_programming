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

inline double peak_compute_flops_per_sm_per_cycle(
    const cudaDeviceProp& prop,
    ComputeRoof roof)
{
    // A100: sm_80
    if (prop.major == 8 && prop.minor == 0) {
        if (roof == ComputeRoof::FP32CudaCores) {
            return 128.0;   // 64 FP32 lanes/SM * 2 FLOPs/FMA
        }
        else if (roof == ComputeRoof::FP16TensorCores) {
            return 2048.0;  // A100 dense FP16 tensor core roof
        }
    }

    std::fprintf(stderr,
                 "Unsupported roofline config for compute capability %d.%d\n",
                 prop.major, prop.minor);
    std::exit(1);
}

inline void printRooflineStats(double flops,
                               double theoretical_bytes_transferred,
                               double kernel_time_ms,
                               const Config& config)
{
    const double kernel_time_s = kernel_time_ms / 1000.0;

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    const double peak_bandwidth_gb_s =
        2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1e6;

    const double peak_bandwidth_bytes_s = peak_bandwidth_gb_s * 1e9;

    const double flops_per_sm_per_cycle =
        peak_compute_flops_per_sm_per_cycle(prop, config.compute_roof);

    const double sm_clock_hz = double(prop.clockRate) * 1000.0;

    const double peak_compute_flops_s =
        double(prop.multiProcessorCount) *
        flops_per_sm_per_cycle *
        sm_clock_hz;

    const double arithmetic_intensity =
        flops / theoretical_bytes_transferred;

    const double peak_memory_flops_s =
        arithmetic_intensity * peak_bandwidth_bytes_s;

    const double peak_roofline_flops_s =
        std::min(peak_compute_flops_s, peak_memory_flops_s);

    const double achieved_flops_s = flops / kernel_time_s;

    const double achieved_tflops_s = achieved_flops_s / 1e12;
    const double peak_roofline_tflops_s = peak_roofline_flops_s / 1e12;
    const double efficiency = achieved_flops_s / peak_roofline_flops_s;

    std::printf("\n--- Roofline Summary ---\n");
    std::printf("Achieved:             %.2f TFLOP/s\n", achieved_tflops_s);
    std::printf("Peak (roofline):      %.2f TFLOP/s\n", peak_roofline_tflops_s);
    std::printf("Efficiency:           %.2f%%\n", 100.0 * efficiency);
    std::printf("-------------------------\n");
}