#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "load_helpers.cuh"
#include "compute_helpers.cuh"
#include "store_helpers.cuh"
#include "kernels.cuh"

__global__
void GEMMSubTiling(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C)
{
    __shared__ float ATile[SUBTILE][SUBTILE + 1];
    __shared__ float BTile[SUBTILE][SUBTILE + 1];

    int startRow = blockIdx.y * SUBTILE;
    int startCol = blockIdx.x * SUBTILE;

    FakeSwizzleParams params = make_params<FakeSwizzleParams>();

    float sum[SUB][SUB];
    #pragma unroll
    for (int i = 0; i < SUB; i++) {
        #pragma unroll
        for (int j = 0; j < SUB; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < K; chunk += SUBTILE) {
        load_with_params(
            params,
            A, ATile,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk
        );
        __syncthreads();

        int kmax = min(SUBTILE, K - chunk);
        compute_subtile(
            ATile, BTile,
            K, kmax,
            sum,
            params
        );
        __syncthreads();
    }

    store_subtile_vec4(
        sum, C, M, N,
        startRow, startCol,
        params.threadRowTile, params.threadColTile,
        alpha, beta
    );
}





