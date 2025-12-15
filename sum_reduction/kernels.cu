#include <cuda_runtime.h>
#include "kernels.cuh"


__global__ void SumReductionBaseline(const float* input,
                                  float* blockResults,
                                  int N)
{


    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    for (int i = tid; i < N; i += stride) {
        sum += input[i];
                                  }
    // Shared Memory for within a block
    extern __shared__ float smem[];
    smem[threadIdx.x] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (threadIdx.x < s){
            smem[threadIdx.x]  += smem[threadIdx.x + s];
        }
    }

    // Write to Block Results
    if (threadIdx.x == 0){
        blockResults[blockIdx.x] = smem[0];
}
}



__global__ void SumReductionWarpShuffle(const float* input,
                                  float* blockResults,
                                  int N)
{


    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    for (int i = tid; i < N; i += stride) {
        sum += input[i];
                                  }
    // Shared Memory for within a block
    extern __shared__ float smem[];
    smem[threadIdx.x] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s >= 32; s >>= 1){
        if (threadIdx.x < s){
            smem[threadIdx.x]  += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    unsigned mask = 0xffffffff;
    sum = smem[threadIdx.x];
    if (threadIdx.x < 32){
        sum += __shfl_down_sync(mask, sum, 16);
        sum += __shfl_down_sync(mask, sum, 8);
        sum += __shfl_down_sync(mask, sum, 4);
        sum += __shfl_down_sync(mask, sum, 2);
        sum += __shfl_down_sync(mask, sum, 1);
    }

    // Write to Block Results
    if (threadIdx.x == 0){
        blockResults[blockIdx.x] = sum;
    }
}



__global__ void SumReductionVec4WarpShuffle(const float* input,
                                  float* blockResults,
                                  int N)
{

    int vecElements = N / 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* input4 = reinterpret_cast<const float4*>(input);

    float sum = 0.0f;

    #pragma unroll 8
    for (int i = tid; i < vecElements; i += stride) {
        float4 c = input4[i];
        sum += c.x + c.y + c.z + c.w;
                                  }

    int leftoverStart = vecElements * 4;
    for (int i = leftoverStart + tid; i < N; i += stride) {
        sum += input[i];
    }

    // Shared Memory for within a block
    extern __shared__ float smem[];
    smem[threadIdx.x] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s >= 32; s >>= 1){
        if (threadIdx.x < s){
            smem[threadIdx.x]  += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    unsigned mask = 0xffffffff;
    sum = smem[threadIdx.x];
    if (threadIdx.x < 32){
        sum += __shfl_down_sync(mask, sum, 16);
        sum += __shfl_down_sync(mask, sum, 8);
        sum += __shfl_down_sync(mask, sum, 4);
        sum += __shfl_down_sync(mask, sum, 2);
        sum += __shfl_down_sync(mask, sum, 1);
    }

    // Write to Block Results
    if (threadIdx.x == 0){
        blockResults[blockIdx.x] = sum;
    }
}