#pragma once
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include "config.h"
#include "param_init.cuh"


__device__ __forceinline__
void load_vec4_or_scalar_to_shared(const float* __restrict__ src,
                                   int row, int col, int ld,
                                   int rowBound, int colBound,
                                   float* dst, int dstCol)
{
    int idx = row * ld + col;

    if (row < rowBound && col + 3 < colBound && ((idx & 3) == 0)) {
        float4 tmp = reinterpret_cast<const float4*>(&src[idx])[0];
        dst[dstCol + 0] = tmp.x;
        dst[dstCol + 1] = tmp.y;
        dst[dstCol + 2] = tmp.z;
        dst[dstCol + 3] = tmp.w;
    } else {
        dst[dstCol + 0] = (row < rowBound && col + 0 < colBound) ? src[idx + 0] : 0.0f;
        dst[dstCol + 1] = (row < rowBound && col + 1 < colBound) ? src[idx + 1] : 0.0f;
        dst[dstCol + 2] = (row < rowBound && col + 2 < colBound) ? src[idx + 2] : 0.0f;
        dst[dstCol + 3] = (row < rowBound && col + 3 < colBound) ? src[idx + 3] : 0.0f;
    }
}

__device__ __forceinline__
void load_vec4_or_scalar_to_shared_transposed(
    const float* __restrict__ src,
    int row, int col, int ld,
    int rowBound, int colBound,
    float* dst, int dst_ld,   // full shared mem pointer + leading dim
    int rowTile, int colTile)
{
    int idx = row * ld + col;

    if (row < rowBound && col + 3 < colBound && ((idx & 3) == 0)) {
        float4 tmp = reinterpret_cast<const float4*>(&src[idx])[0];

        dst[(colTile + 0) * dst_ld + rowTile] = tmp.x;
        dst[(colTile + 1) * dst_ld + rowTile] = tmp.y;
        dst[(colTile + 2) * dst_ld + rowTile] = tmp.z;
        dst[(colTile + 3) * dst_ld + rowTile] = tmp.w;
    } else {
        dst[(colTile + 0) * dst_ld + rowTile] =
            (row < rowBound && col + 0 < colBound) ? src[idx + 0] : 0.0f;

        dst[(colTile + 1) * dst_ld + rowTile] =
            (row < rowBound && col + 1 < colBound) ? src[idx + 1] : 0.0f;

        dst[(colTile + 2) * dst_ld + rowTile] =
            (row < rowBound && col + 2 < colBound) ? src[idx + 2] : 0.0f;

        dst[(colTile + 3) * dst_ld + rowTile] =
            (row < rowBound && col + 3 < colBound) ? src[idx + 3] : 0.0f;
    }
}


__device__ __forceinline__
void load_vec4_or_scalar_to_shared_pad_free(const float* __restrict__ src,
                                   int row, int col, int ld,
                                   int rowBound, int colBound,
                                   float* dst, int dstCol)
{
    int idx = row * ld + col;

    if (row < rowBound &&
        col + 3 < colBound &&
        ((idx & 3) == 0) &&
        ((dstCol & 3) == 0))
    {
        float4 tmp = reinterpret_cast<const float4*>(&src[idx])[0];
        reinterpret_cast<float4*>(&dst[dstCol])[0] = tmp;
    } else {
        dst[dstCol + 0] =
            (row < rowBound && col + 0 < colBound) ? src[idx + 0] : 0.0f;
        dst[dstCol + 1] =
            (row < rowBound && col + 1 < colBound) ? src[idx + 1] : 0.0f;
        dst[dstCol + 2] =
            (row < rowBound && col + 2 < colBound) ? src[idx + 2] : 0.0f;
        dst[dstCol + 3] =
            (row < rowBound && col + 3 < colBound) ? src[idx + 3] : 0.0f;
    }
}




__device__ __forceinline__
void load_vec4_or_scalar_to_shared_async(
    const float* __restrict__ src,
    int row, int col, int ld,
    int rowBound, int colBound,
    float* dst,
    int dstCol)
{
    int idx = row * ld + col;

    bool inBounds =
        (row < rowBound) &&
        (col + 3 < colBound);

    bool srcAligned =
        ((reinterpret_cast<uintptr_t>(&src[idx]) & 0xF) == 0);

    bool dstAligned =
        ((reinterpret_cast<uintptr_t>(&dst[dstCol]) & 0xF) == 0);

    if (inBounds && srcAligned && dstAligned) {
        __pipeline_memcpy_async(
            &dst[dstCol],
            &src[idx],
            sizeof(float4)
        );
    } else {
        dst[dstCol + 0] =
            (row < rowBound && col + 0 < colBound) ? src[idx + 0] : 0.0f;
        dst[dstCol + 1] =
            (row < rowBound && col + 1 < colBound) ? src[idx + 1] : 0.0f;
        dst[dstCol + 2] =
            (row < rowBound && col + 2 < colBound) ? src[idx + 2] : 0.0f;
        dst[dstCol + 3] =
            (row < rowBound && col + 3 < colBound) ? src[idx + 3] : 0.0f;
    }
}

