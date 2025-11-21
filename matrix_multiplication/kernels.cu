#include <cuda_runtime.h>
#include "kernels.cuh"


// GEMM with baseline kernel
__global__ void GEMMBaseline(int M, int N, int K,
                           float alpha,
                           const float* __restrict__ A,  // (M x K)
                           const float* __restrict__ B,  // (K x N)
                           float beta,
                           float* __restrict__ C)  // (M x N))
                           {
    // Global row and column indices to watch linear warp scheduler
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-wide stride
    int strideRow = blockDim.y * gridDim.y;
    int strideCol = blockDim.x * gridDim.x;


    for (int i = row; i < M; i += strideRow) {
        for (int j = col; j < N ; j += strideCol){
            float sum = 0.0f;
            for (int k = 0; k < K; k++){
                sum = fmaf(A[i * K + k], B[k * N + j], sum); 
            }
            //Wriite back to C
            int idx = i * N + j;
            C[idx] = alpha * sum + beta * C[idx];
        }
       
    }

}

// GEMM with Tiling
__global__ void GEMMTiling(int M, int N, int K,
                           float alpha,
                           const float* __restrict__ A,  // (M x K)
                           const float* __restrict__ B,  // (K x N)
                           float beta,
                           float* __restrict__ C)
{
    __shared__ float ATile[TILE][TILE];
    __shared__ float BTile[TILE][TILE];



    // Grid-wide stride
    int strideRow = TILE * gridDim.y;
    int strideCol = TILE * gridDim.x;


    for (int startRow = blockIdx.y * TILE; startRow < M; startRow += strideRow) {
        for (int startCol = blockIdx.x * TILE; startCol < N; startCol += strideCol){

            //Each thread computes one element in the C[threadRowGlobal, threadColGlobal] sub-matrix 
            int threadRowGlobal = startRow + threadIdx.y;
            int threadColGlobal = startCol + threadIdx.x;

            float sum = 0.0f;

            int rowAGlobal = threadRowGlobal;
            int colBGlobal = threadColGlobal;
            for (int chunk = 0; chunk < K; chunk += TILE){         
                int colAGlobal = threadIdx.x + chunk;
                int rowBGlobal = threadIdx.y + chunk;

                //Loads into shared memory, in 
                ATile[threadIdx.y][threadIdx.x] = (rowAGlobal < M && colAGlobal < K) ? A[rowAGlobal * K + colAGlobal] : 0.0f;
                BTile[threadIdx.y][threadIdx.x] = (rowBGlobal < K && colBGlobal < N) ? B[rowBGlobal * N + colBGlobal] : 0.0f;

                __syncthreads();

                // Continue accumulation
                int kmax = min(TILE, K-chunk);
                
                #pragma unroll
                for (int k = 0; k < kmax; k++){
                    sum = fmaf(ATile[threadIdx.y][k], BTile[k][threadIdx.x], sum); 
                }
                __syncthreads();
        }
        //Write back to memory
        if (threadRowGlobal < M && threadColGlobal < N) {
                int idx = threadRowGlobal * N + threadColGlobal;
                float COld = (beta != 0.0f) ? C[idx] : 0.0f;
                C[idx] = alpha * sum + beta *COld;
        }
    }
       
    }
}



