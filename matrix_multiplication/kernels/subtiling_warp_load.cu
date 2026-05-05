#include <cuda_runtime.h>
#include <cstdio>
#include "config.h"
#include "kernels.cuh"
#include "param_init.cuh"
#include "load_helpers.cuh"
#include "compute_helpers.cuh"
#include "store_helpers.cuh"




__device__ __forceinline__
void load_subtile_linear_slab(const float* __restrict__ A,
                       float ATile[SUBTILE][SUBTILE+PADDING],
                       const float* __restrict__ B,
                       float BTile[SUBTILE][SUBTILE+PADDING],
                       int M, int K, int N,
                       int startRow, int startCol,
                       int chunk,
                       const SlabParams& params)
{   
    constexpr int SLAB_DIM_ROWS = (SUBTILE + 3) >> 2;
    constexpr int SLAB_DIM_COLS = (SUBTILE + 31) >> 5;
    constexpr int TOTAL_SLABS   = SLAB_DIM_ROWS * SLAB_DIM_COLS;

    int threadRowGlobalOriginA = startRow;
    int threadColGlobalOriginA = chunk;

    int threadRowGlobalOriginB = chunk;
    int threadColGlobalOriginB = startCol;

    for (int slabNum = params.warpId; slabNum < TOTAL_SLABS; slabNum += params.numWarps){ // loop over all slabs
        int slabRowStart = slabNum / SLAB_DIM_COLS; // map back to the starting row of tile for that slab
        int slabColStart = slabNum % SLAB_DIM_COLS; // map back to starting col of tile for that slab
        int rowTile = 4 * slabRowStart + params.slabRowIdx; // tells us which row of the tile the thread is working on
        int colTile = 32 * slabColStart + 4 * params.slabColIdx; // tells us which col of the tile the thread is starting on

        //Load in A
        int rowA = threadRowGlobalOriginA + rowTile;
        int colA = threadColGlobalOriginA + colTile;

        load_vec4_or_scalar_to_shared(A, rowA, colA, K,
                                    M, K, &ATile[rowTile][0], colTile);



        //Load in B
        int rowB = threadRowGlobalOriginB + rowTile;
        int colB = threadColGlobalOriginB + colTile;
        load_vec4_or_scalar_to_shared(B, rowB, colB, N,
                                    K, N, &BTile[rowTile][0], colTile);
        
    }


}


__global__
void GEMMSubTilingLoadSlabLinear(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C)
{
    __shared__ float ATile[SUBTILE][SUBTILE + PADDING];
    __shared__ float BTile[SUBTILE][SUBTILE + PADDING];

    int startRow = blockIdx.y * SUBTILE;
    int startCol = blockIdx.x * SUBTILE;

    SlabParams params = make_linear_slab_params();

    float sum[SUB][SUB];
    #pragma unroll
    for (int i = 0; i < SUB; i++) {
        #pragma unroll
        for (int j = 0; j < SUB; j++) {
            sum[i][j] = 0.0f;
        }
    }

    for (int chunk = 0; chunk < K; chunk += SUBTILE) {
        load_subtile_linear_slab(
            A, ATile,
            B, BTile,
            M, K, N,
            startRow, startCol,
            chunk, params
        );
        __syncthreads();

        int kmax = min(SUBTILE, K - chunk);
        compute_subtile(
            ATile, BTile,
            kmax,
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


