#include <cuda_runtime.h>
#include "kernels.cuh"

__global__ void Transpose(const float* __restrict__ input,
                              float* __restrict__ output,
                              int width,
                              int height)
{
   int row = blockDim.y * blockIdx.y + threadIdx.y;
   int col = blockDim.x * blockIdx.x + threadIdx.x;


   __shared__ float tile[TILE][TILE + 1];

   
   //Load data into shared memory tile 
   if (row < height && col < width){
      tile[threadIdx.y][threadIdx.x] = input[row * width + col];

   }
   
   int width_output = height;
   int height_output = width;

   int out_row = blockDim.x * blockIdx.x + threadIdx.y;
   int out_col = blockDim.y * blockIdx.y + threadIdx.x;
   
   __syncthreads();
   
  if (out_col < width_output && out_row < height_output){
      output[out_row * width_output + out_col] = tile[threadIdx.x][threadIdx.y]
  }      
         


}