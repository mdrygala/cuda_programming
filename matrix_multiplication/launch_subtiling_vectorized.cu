#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <stdexcept>

#include "config.h"
#include "kernels/kernels.cuh"
#include "utils/launch_utils.cuh"

void set_block_and_grid(dim3& block, dim3& grid, int M, int N)
{
    constexpr int THREADS_M = TILE_REGISTER_VEC / THREAD_DIM_REGISTER_VEC;
    constexpr int THREADS_N = TILE_REGISTER_VEC / THREAD_DIM_REGISTER_VEC;

    set_block_and_grid_subtile<
        TILE_REGISTER_VEC,
        TILE_REGISTER_VEC,
        THREADS_N,
        THREADS_M
    >(block, grid, M, N);
}

template <typename InputT>
void launch_kernel(int M, int N, int K,
                   float alpha,
                   const InputT* __restrict__ dA,
                   const InputT* __restrict__ dB,
                   float beta,
                   float* __restrict__ dC,
                   dim3& grid,
                   dim3& block);

template <>
void launch_kernel<float>(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ dA,
                          const float* __restrict__ dB,
                          float beta,
                          float* __restrict__ dC,
                          dim3& grid,
                          dim3& block)
{
    GEMMSubTilingVec4<<<grid, block>>>(
        M, N, K,
        alpha,
        dA, dB,
        beta,
        dC
    );
}

int main(int argc, char** argv)
{
    Config config;
    parseArgs(argc, argv, config);

    if (config.data_type != DataType::Float32) {
        throw std::runtime_error("launch_subtiling_vectorized only supports float");
    }

    config.compute_roof = ComputeRoof::FP32CudaCores;

    run_verification<float>(256, 256, 256, config);
    run_profile<float>(1 << 12, 1 << 12, 1 << 12, config);

    return 0;
}