#pragma once
#include <cuda_runtime.h>
#include "config.h"


__device__ __forceinline__
void store_subtile_scalar(float sum[SUB][SUB],
                         float*  __restrict__ C, int M, int N, 
                         int startRow, int startCol,
                         int threadRowTile, int threadColTile,
                        float alpha, float beta)
{
    int threadRowGlobalOrigin = startRow + threadRowTile;
    int threadColGlobalOrigin = startCol + threadColTile;
    #pragma unroll
    for (int i = 0; i < SUB; i++){
        int r = threadRowGlobalOrigin + i;
        if (r >= M) break;
        #pragma unroll
        for (int j = 0; j < SUB; j++){
            int c = threadColGlobalOrigin + j;
            if (c >= N) break;
            int idx = r * N + c;
            float cold = (beta != 0.0f) ? C[idx] : 0.0f;
            C[idx] = alpha * sum[i][j] + beta * cold;
        }
    }
}

__device__ __forceinline__
void store_subtile_vec4(float sum[SUB][SUB],
                         float*  __restrict__ C, int M, int N, 
                         int startRow, int startCol,
                         int threadRowTile, int threadColTile,
                        float alpha, float beta)
{
    int threadRowGlobalOrigin = startRow + threadRowTile;
    int threadColGlobalOrigin = startCol + threadColTile;

    // Fast path when beta == 0: no read-modify-write needed
    if (beta == 0.0f) {
        #pragma unroll
        for (int i = 0; i < SUB; i++) {
            int r = threadRowGlobalOrigin + i;
            if (r >= M) break;

            int base = r * N + threadColGlobalOrigin;

            // vector part: j in steps of 4
            #pragma unroll
            for (int j = 0; j + 3 < SUB; j += 4) {
                int c = threadColGlobalOrigin + j;
                if (c + 3 >= N) break;

                int idx = base + j;
                // alignment check for float4 store
                if (((idx & 3) == 0)) {
                    float4 out;
                    out.x = alpha * sum[i][j + 0];
                    out.y = alpha * sum[i][j + 1];
                    out.z = alpha * sum[i][j + 2];
                    out.w = alpha * sum[i][j + 3];
                    reinterpret_cast<float4*>(C)[idx >> 2] = out;
                } else {
                    // unaligned fallback
                    C[idx + 0] = alpha * sum[i][j + 0];
                    C[idx + 1] = alpha * sum[i][j + 1];
                    C[idx + 2] = alpha * sum[i][j + 2];
                    C[idx + 3] = alpha * sum[i][j + 3];
                }
            }

            // tail (or if N bound cuts vector loop early)
            #pragma unroll
            for (int j = (SUB & ~3); j < SUB; j++) {
                int c = threadColGlobalOrigin + j;
                if (c >= N) break;
                int idx = base + j;
                C[idx] = alpha * sum[i][j];
            }
        }
        return;
    }

    // General path beta != 0: need to read C (vectorize load+store when possible)
    #pragma unroll
    for (int i = 0; i < SUB; i++) {
        int r = threadRowGlobalOrigin + i;
        if (r >= M) break;

        int base = r * N + threadColGlobalOrigin;

        #pragma unroll
        for (int j = 0; j + 3 < SUB; j += 4) {
            int c = threadColGlobalOrigin + j;
            if (c + 3 >= N) break;

            int idx = base + j;

            if (((idx & 3) == 0)) {
                float4 cold = reinterpret_cast<const float4*>(C)[idx >> 2];
                float4 out;
                out.x = alpha * sum[i][j + 0] + beta * cold.x;
                out.y = alpha * sum[i][j + 1] + beta * cold.y;
                out.z = alpha * sum[i][j + 2] + beta * cold.z;
                out.w = alpha * sum[i][j + 3] + beta * cold.w;
                reinterpret_cast<float4*>(C)[idx >> 2] = out;
            } else {
                // unaligned fallback
                float c0 = C[idx + 0];
                float c1 = C[idx + 1];
                float c2 = C[idx + 2];
                float c3 = C[idx + 3];
                C[idx + 0] = alpha * sum[i][j + 0] + beta * c0;
                C[idx + 1] = alpha * sum[i][j + 1] + beta * c1;
                C[idx + 2] = alpha * sum[i][j + 2] + beta * c2;
                C[idx + 3] = alpha * sum[i][j + 3] + beta * c3;
            }
        }

        // scalar tail
        #pragma unroll
        for (int j = (SUB & ~3); j < SUB; j++) {
            int c = threadColGlobalOrigin + j;
            if (c >= N) break;
            int idx = base + j;
            float cold = C[idx];
            C[idx] = alpha * sum[i][j] + beta * cold;
        }
    }
}



struct StoreScalar {
    __device__ __forceinline__
    static void run(
        float sum[SUB][SUB],
        float*  __restrict__ C, int M, int N, 
        int startRow, int startCol,
        int threadRowTile, int threadColTile,
        float alpha, float beta
    ) {
        store_subtile_scalar(
            sum, C, M, N, startRow, startCol,
            threadRowTile, threadColTile, alpha, beta
        );
    }
};

struct StoreVec4 {
    __device__ __forceinline__
    static void run(
        float sum[SUB][SUB],
        float*  __restrict__ C, int M, int N, 
        int startRow, int startCol,
        int threadRowTile, int threadColTile,
        float alpha, float beta
    ) {
        store_subtile_vec4(
            sum, C, M, N, startRow, startCol,
            threadRowTile, threadColTile, alpha, beta
        );
    }
};