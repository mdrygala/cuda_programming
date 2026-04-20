#pragma once
#include <cuda_runtime.h>
#include "config.h"

// Non-templated kernels (decl only)
__global__ void GEMMBaseline(int M,int N,int K,
                             float alpha,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float beta,
                             float* __restrict__ C);

__global__ void GEMMTiling(int M,int N,int K,
                           float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B,
                           float beta,
                           float* __restrict__ C);

// Wrapper: one named subtile kernel you can profile/call easily

__global__ void GEMMSubtileRegNaive(int M,int N,int K,
                                 float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C);

__global__ void GEMMSubtileRegVec4(int M,int N,int K,
                                 float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C);

__global__ void GEMMSubtileWarpSlab(int M,int N,int K,
                                 float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C);


__global__ void GEMMSubtileSwzl(int M,int N,int K,
                                 float alpha,
                                 const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float beta,
                                 float* __restrict__ C);
