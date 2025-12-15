#include <cuda_runtime.h>
#include "kernels.cuh"

__global__ void SpMVBaseline(const float* __restrict__ values,
                             const int* __restrict__ col_idx,
                             const int* __restrict__ row_ptr,
                             const float* __restrict__ x,
                             float* __restrict__ y,
                             int M){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    
    if(tid < M){
        for (int k = row_ptr[tid]; k < row_ptr[tid + 1]; k++){
            sum += values[k] * x[col_idx[k]];
        }
        y[tid] = sum;
    }
    
}


__global__ void SpMVWarpRow(const float* __restrict__ values,
                             const int* __restrict__ col_idx,
                             const int* __restrict__ row_ptr,
                             const float* __restrict__ x,
                             float* __restrict__ y,
                             int M{
     
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int warpId = tid >> 5;

int lane = threadIdx.x & 31;

float sum = 0.0f;

if(warpId < M){
    for(int k = row_ptr[warpId] + lane; k < row_ptr[warpId+1]; k+=32){
        sum += values[k] * x[col_idx[k]];
    }
    unsigned mask = __activemask();
    sum += __shfl_down_sync(mask, sum, 16);
    sum += __shfl_down_sync(mask, sum, 8);
    sum += __shfl_down_sync(mask, sum, 4);
    sum += __shfl_down_sync(mask, sum, 2);
    sum += __shfl_down_sync(mask, sum, 1);

    if (lane == 0){
        y[warpId] = sum;
    }
}


}