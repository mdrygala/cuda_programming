#pragma once
#include <cuda_runtime.h>

#ifndef TILE_REGISTER_VEC
#define TILE_REGISTER_VEC 64
#endif

#ifndef THREAD_DIM_REGISTER_VEC
#define THREAD_DIM_REGISTER_VEC 4
#endif

#ifndef PADDING_REGISTER_VEC
#define PADDING_REGISTER_VEC 0
#endif

__global__  
void GEMMSubTilingVec4(int M, int N, int K,
                          float alpha,
                          const float* __restrict__ A,
                          const float* __restrict__ B,
                          float beta,
                          float* __restrict__ C);




