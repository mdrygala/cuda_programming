#pragma once
#include <cuda_runtime.h>
#include "../config.h"

#include "../param_init.cuh"
#include "../load_helpers.cuh"
#include "../compute_helpers.cuh"
#include "../store_helpers.cuh"


// Non-templated kernels (decl only)
__global__ void GEMMBaseline(int M,int N,int K,
                             float alpha,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float beta,
                             float* __restrict__ C);
__global__ void GEMMTiling(int M, int N, int K,
                           float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B,
                           float beta,
                           float* __restrict__ C);
                           
__global__
void GEMMSubTilingScalar(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C);

__global__                          
void GEMMSubTilingScalarTransposed(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C);

__global__  
void GEMMSubTilingVec4(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C);

__global__
void GEMMSubTilingVec4Transposed(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C);
__global__                           
void GEMMSubTilingLoadSlabLinear(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C);
__global__                           
void GEMMSubTilingLoadSlab2D(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C);
__global__
void GEMMSubTilingLoadSlabLinearTransposed(int M, int N, int K,
                                           float alpha,
                                           const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float beta,
                                           float* __restrict__ C);

__global__
void GEMMSubTilingLoadSlabGenDims(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C);
__global__
void GEMMSubTilingLoadSlabGenDimsDoubleBuffered(int M, int N, int K,
                                                float alpha,
                                                const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float beta,
                                                float* __restrict__ C);
__global__
void GEMMSubTilingLoadSlabLinearAsync(int M, int N, int K,
                                      float alpha,
                                      const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float beta,
                                      float* __restrict__ C);