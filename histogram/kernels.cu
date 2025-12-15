#include <cuda_runtime.h>
#include "kernels.cuh"


__global__ void histogramBaseline(const int* __restrict__ x, int* bins, int N){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i+=stride){
        int val = x[i];
        atomicAdd(&bins[val]);
    }

}

__global__ void histogramBlockShare(const int* __restrict__ x, int* bins, int N, int B){

    extern __shared__ int smem[];

    for (int i=threadIdx.x; i < B; i +=blockDim.x){
        smem[i]=0;
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __syncthreads();

    for (int i=tid; i < N; i += stride){
        int val = x[i];
        atomicAdd(&smem[val], 1);
    }
    __syncthreads();

    for (int i=threadIdx.x; i < B; i +=blockDim.x){
        atomicAdd(&bins[i], smem[i]);
    }

}


__global__ void histogramWarpPrivate(const int* __restrict__ x, int* bins, int N, int B){

    extern __shared__ int smem[];

    int numWarps = blockDim.x >> 5;
    int warpId = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    for (int i=lane; i < B; i +=32){
        smem[B*warpId + i]=0;
    }

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __syncthreads();

    for (int i=tid; i < N; i += stride){
        int val = x[i];
        atomicAdd(&smem[B * warpId + val], 1);
    }

    

    for (int s = numWarps/2; s>0; s>>=1){
        if (warpId < s){
            for (int i=lane; i < B; i +=32){
                smem[B*warpId + i] += smem[B*(warpId + s) + i];

        }  
    }
    __syncthreads();
    }
    
    __syncthreads();
    for (int i=threadIdx.x; i < B; i +=blockDim.x){
        atomicAdd(&bins[i], smem[i]);
    }

}