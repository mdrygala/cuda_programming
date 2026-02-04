#include <cuda_runtime.h>
#include "gemm_config.h"
#include "gemm_loaders.cuh"
#include "gemm_compute.cuh"
#include "kernels.cuh"          // include declarations

// -------------------- Baseline --------------------
__global__ void GEMMBaseline(int M, int N, int K,
                             float alpha,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float beta,
                             float* __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++){
        sum = fmaf(A[row * K + k], B[k * N + col], sum);
    }

    int idx = row * N + col;
    float cold = (beta != 0.0f) ? C[idx] : 0.0f;
    C[idx] = alpha * sum + beta * cold;   // FIX: don’t read C twice
}

// -------------------- Tiling --------------------
__global__ void GEMMTiling(int M, int N, int K,
                           float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B,
                           float beta,
                           float* __restrict__ C)
{
    __shared__ float ATile[TILE][TILE];
    __shared__ float BTile[TILE][TILE];

    int startRow = blockIdx.y * TILE;
    int startCol = blockIdx.x * TILE;

    int threadRowGlobal = startRow + threadIdx.y;
    int threadColGlobal = startCol + threadIdx.x;

    float sum = 0.0f;

    for (int chunk = 0; chunk < K; chunk += TILE){
        int rowA = threadRowGlobal;
        int colA = chunk + threadIdx.x;

        int rowB = chunk + threadIdx.y;
        int colB = threadColGlobal;

        ATile[threadIdx.y][threadIdx.x] =
            (rowA < M && colA < K) ? A[rowA * K + colA] : 0.0f;

        BTile[threadIdx.y][threadIdx.x] =
            (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;

        __syncthreads();

        int kmax = min(TILE, K - chunk);
        #pragma unroll
        for (int k = 0; k < kmax; k++){
            sum = fmaf(ATile[threadIdx.y][k], BTile[k][threadIdx.x], sum);
        }
        __syncthreads();
    }

    if (threadRowGlobal < M && threadColGlobal < N){
        int idx = threadRowGlobal * N + threadColGlobal;
        float cold = (beta != 0.0f) ? C[idx] : 0.0f;
        C[idx] = alpha * sum + beta * cold;
    }
}

// -------------------- Subtiling template (definition stays in .cu) --------------------
template <typename LoaderFunc, typename ComputeFunc>
__global__ void GEMMSubTiling(int M, int N, int K,
                              float alpha,
                              const float* __restrict__ A,
                              const float* __restrict__ B,
                              float beta,
                              float* __restrict__ C)
{
    __shared__ float ATile[SUBTILE][SUBTILE+1];
    __shared__ float BTile[SUBTILE][SUBTILE+1];

    int strideRow = SUBTILE * gridDim.y;
    int strideCol = SUBTILE * gridDim.x;

    int threadRowTile = threadIdx.y * SUB;
    int threadColTile = threadIdx.x * SUB;

    for (int startRow = blockIdx.y * SUBTILE; startRow < M; startRow += strideRow) {
        for (int startCol = blockIdx.x * SUBTILE; startCol < N; startCol += strideCol) {

            int threadRowGlobalOrigin = startRow + threadRowTile;
            int threadColGlobalOrigin = startCol + threadColTile;

            float sum[SUB][SUB];
            #pragma unroll
            for (int i = 0; i < SUB; i++)
                #pragma unroll
                for (int j = 0; j < SUB; j++)
                    sum[i][j] = 0.0f;

            for (int chunk = 0; chunk < K; chunk += SUBTILE){
                int threadRowGlobalOriginA = threadRowGlobalOrigin;
                int threadColGlobalOriginA = threadColTile + chunk;

                int threadRowGlobalOriginB = threadRowTile + chunk;
                int threadColGlobalOriginB = threadColGlobalOrigin;

                LoaderFunc::run(A, ATile, B, BTile, M, K, N,
                                startRow, startCol, chunk,
                                threadRowTile, threadColTile,
                                threadRowGlobalOriginA, threadColGlobalOriginA,
                                threadRowGlobalOriginB, threadColGlobalOriginB);
                __syncthreads();

                int kmax = min(SUBTILE, K - chunk);
                ComputeFunc::run(ATile, BTile, K, kmax,
                                 sum, threadRowTile, threadColTile);
                __syncthreads();
            }

            #pragma unroll
            for (int i = 0; i < SUB; i++){
                int r = threadRowGlobalOrigin + i;
                if (r >= M) break;
                #pragma unroll
                for (int j = 0; j < SUB; j++){
                    int c = threadColGlobalOrigin + j;
                    if (c >= N) break;
                    int idx = r * N + c;
                    float cold = (beta != 0.0f) ? C[idx] : 0.0f;
                    C[idx] = alpha * sum[i][j] + beta * cold;
                }
            }
        }
    }
}

// -------------------- One concrete subtile kernel name --------------------
// This is what you call/profile: fixed choice of loader+compute.
__global__ void GEMMSubtileFinal(int M,int N,int K,
                                 float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C)
{
    GEMMSubTiling<LoaderVec4Swizzle, ComputeSwizzle>(M, N, K, alpha, A, B, beta, C);
}
