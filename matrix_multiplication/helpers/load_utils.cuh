#pragma once
#include <cuda_runtime.h>
#include "config.h"
#include "param_init.cuh"





template <typename InputT>
__device__ __forceinline__
void load_to_shared(const InputT* __restrict__ src,
                    int row, int col, int ld,
                    int rowBound, int colBound,
                    InputT* dst,
                    int dstCol)
{
    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);

    int idx = row * ld + col;

    if (row < rowBound &&
        col + VEC_ELEMS - 1 < colBound &&
        ((reinterpret_cast<uintptr_t>(&src[idx]) & 0xF) == 0))
    {
        int4 raw = reinterpret_cast<const int4*>(&src[idx])[0];
        InputT* vals = reinterpret_cast<InputT*>(&raw);

#pragma unroll
        for (int v = 0; v < VEC_ELEMS; v++) {
            dst[dstCol + v] = vals[v];
        }
    }
    else {
#pragma unroll
        for (int v = 0; v < VEC_ELEMS; v++) {
            dst[dstCol + v] =
                (row < rowBound && col + v < colBound)
                    ? src[idx + v]
                    : InputT{};
        }
    }
}


template <typename InputT>
__device__ __forceinline__
void load_to_shared_pad_free(const InputT* __restrict__ src,
                             int row, int col, int ld,
                             int rowBound, int colBound,
                             InputT* dst,
                             int dstCol)
{
    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);

    int idx = row * ld + col;

    if (row < rowBound &&
        col + VEC_ELEMS - 1 < colBound &&
        ((reinterpret_cast<uintptr_t>(&src[idx]) & 0xF) == 0) &&
        ((reinterpret_cast<uintptr_t>(&dst[dstCol]) & 0xF) == 0))
    {
        *reinterpret_cast<int4*>(&dst[dstCol]) =
            *reinterpret_cast<const int4*>(&src[idx]);
    }
    else {
#pragma unroll
        for (int v = 0; v < VEC_ELEMS; v++) {
            dst[dstCol + v] =
                (row < rowBound && col + v < colBound)
                    ? src[idx + v]
                    : InputT{};
        }
    }
}


template <typename InputT>
__device__ __forceinline__
void load_to_shared_transposed(
    const InputT* __restrict__ src,
    int row, int col, int ld,
    int rowBound, int colBound,
    InputT* dst,
    int dst_ld,
    int rowTile,
    int colTile)
{
    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);

    int idx = row * ld + col;

    if (row < rowBound &&
        col + VEC_ELEMS - 1 < colBound &&
        ((reinterpret_cast<uintptr_t>(&src[idx]) & 0xF) == 0))
    {
        int4 raw = reinterpret_cast<const int4*>(&src[idx])[0];
        InputT* vals = reinterpret_cast<InputT*>(&raw);

        #pragma unroll
        for (int v = 0; v < VEC_ELEMS; v++) {
            dst[(colTile + v) * dst_ld + rowTile] = vals[v];
        }
    }
    else {
        #pragma unroll
        for (int v = 0; v < VEC_ELEMS; v++) {
            dst[(colTile + v) * dst_ld + rowTile] =
                (row < rowBound && col + v < colBound)
                    ? src[idx + v]
                    : InputT{};
        }
    }
}