#include <cuda_runtime.h>
#include <cuda_fp16.h>        // for __half
#include <cuda_bf16.h>        // for __nv_bfloat16


//B4
__global__ void reduceSum(const float* a, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float* smem[];

    if (i < N){
        smem[i] = a[i];
    }
    __syncthreads();

    if (i < N-1){
        out[i] = smem[i] + smem[i+1];
    }
    
}

//B8

__global__ void example(float* a) {
    __shared__ float temp;
    temp = a[0];
    __syncthreads();
    atomicAdd(&a[0], temp + threadIdx.x);
}

//B9

__global void example(const float* __restrict__ A, float* __restrict__ C, int N){
    int i = blockIdx.x * blockDim.x + threadIdx;

    int c = 1 & threadIdx.x;
    if (i < N){
        C[i] = (2 + c) * A[i];
    }
    
}

//C1 
__shared__ float tile[32][32];
tile[tid][tx] = A[idx];
tile[tid+1][tx] = A[idx+1];
__syncthreads();
B[idx] = tile[tx][tid];


//q1 mixed precision dot product naive:

__global__ void MPDot(const __half* __restrict__ a, const __half* __restrict__ b, float* __restrict__ out, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = gridDim.x * blockDim.x;

    float sum = 0.0f;

    for (int i = tid; i < N; i+= stride){
        sum += __half2float(a[i]) * __half2float(b[i]);
    }

    atomicAdd(out, sum);

}

//q2 mixed precision axpy y = a * x + y:

__global__ void MPAXPY(const __half* __restrict__ x, __half* __restrict__ y, __half a, int N){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float af = __half2float(a);   // convert scalar once

    for (int i = tid; i < N; i+=stride){
        float result = af * __half2float(x[i]) + __half2float(y[i]);
        y[i] = __float2half(result);
    }
}

//q6 mixed precision GEMM naive

__global void MPGEMMNaive(const __half* __restrict__ A, const __half* __restrict__ B,
     __half* __restrict__ C, int M, int N, int K, __half alpha, __half beta){

        int row = blockDim.y * blockIdx.y + threadIdx.y;
        int col = blockDim.x * blockIdx.x + threadIdx.x;

        if (row >= M || col >= M) return;

        float alphaf = __half2float(alpha);
        float betaf = __half2float(beta);

        float acc = 0.0f;

        for (int k = 0; k<K; k++){
            acc += __half2float(A[K*row + k]) * __half2float(B[k*K + col]);
        }
        int idx = row * N + col;
        float result = alphaf * acc + betaf * __half2float(C[idx]);
        C[idx] = __float2half(result);
     }


     __global void GEMMTile(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K, float alpha, float beta){


     __shared__ float ATile[TILE][TILE];
     __shared__ float BTile[TILE][TILE];


     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.y + threadIdx.x;

     int numChunks = (K + TILE - 1)/TILE;

     float sum = 0.0f;

     for (int chunk=0; chunk < numChunks; chunks++){
        int colATile = chunk*TILE + threadIdx.x;
        if(row < M && colATile < K){
            ATile[threadIdx.y][threadIdx.x] = A[row * K + colATile];
        }
        int rowBTile = chunk*TILE + threadIdx.y;
        if (rowBTile < K && col < N){
            BTile[threadIdx.y][threadIdx.x] = B[rowBTile * N + col];
        }
        

        __syncthreads();

        int kmax = std::min(TILE, K-(chunk*TILE));

        for (int k = 0; k<kmax; k++){
            sum += A[threadIdx.y][k] * B[k][threadIdx.x];
        }
        __syncthreads();

     }
     if (row >= M || col >=N) return;
     int idx = row * M + col;
     C[idx] = alpha * sum + beta * C[idx];
     

     }