#pragma once

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 32
#endif

__global__ void SpMVBaseline(const float* __restrict__ values,
                             const int* __restrict__ col_idx,
                             const int* __restrict__ row_ptr,
                             const float* __restrict__ x,
                             float* __restrict__ y,
                             int M);

__global__ void SpMVWarpRow(const float* __restrict__ values,
                             const int* __restrict__ col_idx,
                             const int* __restrict__ row_ptr,
                             const float* __restrict__ x,
                             float* __restrict__ y,
                             int M);