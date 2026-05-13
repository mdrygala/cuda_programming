#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <stdexcept>

#include "config.h"
// #include "kernels/kernels.cuh"
#include "utils/launch_utils.cuh"
#include "kernels/subtiling_vectorized_transposed.cuh"

void set_block_and_grid(dim3& block, dim3& grid, int M, int N)
{
    constexpr int THREADS_M = TILE_REGISTER_VEC_TRANSPOSED_M / THREAD_DIM_REGISTER_VEC_TRANSPOSED;
    constexpr int THREADS_N = TILE_REGISTER_VEC_TRANSPOSED_N / THREAD_DIM_REGISTER_VEC_TRANSPOSED;

    set_block_and_grid_subtile<
        TILE_REGISTER_VEC_TRANSPOSED_M,
        TILE_REGISTER_VEC_TRANSPOSED_N,
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
                   dim3& block){
    static_assert(
        std::is_same<InputT, float>::value ||
        std::is_same<InputT, __half>::value,
        "launch_kernel<InputT>: InputT must be float or __half"
    );
    GEMMSubTilingVec4Transposed<InputT><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    
}



int main(int argc, char** argv) {
    Config config;
    parseArgs(argc, argv, config);

    config.compute_roof = ComputeRoof::FP32CudaCores;

    run_selected_datatype(config);
    

    return 0;
}
