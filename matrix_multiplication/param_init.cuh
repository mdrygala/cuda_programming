#pragma once
#include <cuda_runtime.h>

struct SlabParams {
    int threadRowTile;
    int threadColTile;

    int numWarps;
    int warpId;
    int slabRowIdx;
    int slabColIdx;
};


__device__ __forceinline__
SlabParams make_linear_slab_params()
{
    SlabParams params;

    params.threadRowTile = threadIdx.y * SUB;
    params.threadColTile = threadIdx.x * SUB;


    params.numWarps = (blockDim.x * blockDim.y) >> 5;

    int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;
    params.warpId = threadBlockIdx >> 5;

    int laneId = threadBlockIdx & 31;
    params.slabRowIdx = laneId >> 3;
    params.slabColIdx = laneId & 7;

    return params;
}

struct SlabParamsLinearTransposed {
    int threadRowTile;
    int threadColTile;

    int warpId;
    int laneId;
    int numWarps;
};


__device__ __forceinline__
SlabParamsLinearTransposed make_slab_params_linear_transposed()
{
    SlabParamsLinearTransposed params;

    params.threadRowTile = threadIdx.y * SUB;
    params.threadColTile = threadIdx.x * SUB;

    int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;

    params.warpId   = threadBlockIdx >> 5;
    params.laneId   = threadBlockIdx & 31;
    params.numWarps = (blockDim.x * blockDim.y) >> 5;

    return params;
}

struct SlabParamsGenDim {
    int threadRowTile;
    int threadColTile;

    int numWarps;
    int warpId;

    int slabRowIdxA;
    int slabColIdxA;

    int slabRowIdxB;
    int slabColIdxB;
};

__device__ __forceinline__
SlabParamsGenDim make_linear_slab_params_gendim()
{
    SlabParamsGenDim params;

    params.threadRowTile = threadIdx.y * SUB;
    params.threadColTile = threadIdx.x * SUB;

    int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int laneId = threadBlockIdx & 31;

    params.numWarps = (blockDim.x * blockDim.y) >> 5;
    params.warpId   = threadBlockIdx >> 5;

    constexpr int VEC = 4;
    constexpr int WARP_SIZE = 32;

    static_assert(SUBTILE_K  % VEC == 0, "SUBTILE_K must be divisible by 4");
    static_assert(SUBTILE_MN % VEC == 0, "SUBTILE_MN must be divisible by 4");

    constexpr int A_VEC_COLS = SUBTILE_K  / VEC;
    constexpr int B_VEC_COLS = SUBTILE_MN / VEC;

    static_assert(WARP_SIZE % A_VEC_COLS == 0,
                  "SUBTILE_K / 4 must divide 32");
    static_assert(WARP_SIZE % B_VEC_COLS == 0,
                  "SUBTILE_MN / 4 must divide 32");

    params.slabRowIdxA = laneId / A_VEC_COLS;
    params.slabColIdxA = laneId % A_VEC_COLS;

    params.slabRowIdxB = laneId / B_VEC_COLS;
    params.slabColIdxB = laneId % B_VEC_COLS;

    return params;
}