#pragma once
#include <cuda_runtime.h>

struct SlabParams {
    int threadRowTile;
    int threadColTile;

    int warpId;
    int slabRowIdx;
    int slabColIdx;
};

template <typename InputT>
__device__ __forceinline__
SlabParams make_linear_slab_params()
{
    SlabParams params;

    params.threadRowTile = threadIdx.y * SUB;
    params.threadColTile = threadIdx.x * SUB;

    int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;
    params.warpId = threadBlockIdx >> 5;

    int laneId = threadBlockIdx & 31;

    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);
    constexpr int VEC_COLS_PER_SLAB = 32 / VEC_ELEMS;

    params.slabRowIdx = laneId / VEC_COLS_PER_SLAB;
    params.slabColIdx = laneId % VEC_COLS_PER_SLAB;

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

    int warpId;

    int slabRowIdxA;
    int slabColIdxA;

    int slabRowIdxB;
    int slabColIdxB;
};

template <typename InputT>
__device__ __forceinline__
SlabParamsGenDim make_linear_slab_params_gendim()
{ 
    SlabParamsGenDim params;

    params.threadRowTile = threadIdx.y * SUB;
    params.threadColTile = threadIdx.x * SUB;

    int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;

    params.warpId  = threadBlockIdx >> 5;
    int laneId = threadBlockIdx & 31;

    

    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);
    constexpr int WARP_SIZE = 32;

    static_assert(VEC_BYTES % sizeof(InputT) == 0,
                  "InputT must divide 16 bytes");

    static_assert(SUBTILE_K % VEC_ELEMS == 0,
                  "SUBTILE_K must be divisible by VEC_ELEMS");

    static_assert(SUBTILE_MN % VEC_ELEMS == 0,
                  "SUBTILE_MN must be divisible by VEC_ELEMS");

    constexpr int A_VEC_COLS = SUBTILE_K  / VEC_ELEMS;
    constexpr int B_VEC_COLS = SUBTILE_MN / VEC_ELEMS;

    static_assert(WARP_SIZE % A_VEC_COLS == 0,
                  "SUBTILE_K / VEC_ELEMS must divide 32");

    static_assert(WARP_SIZE % B_VEC_COLS == 0,
                  "SUBTILE_MN / VEC_ELEMS must divide 32");

    params.slabRowIdxA = laneId / A_VEC_COLS;
    params.slabColIdxA = laneId % A_VEC_COLS;

    params.slabRowIdxB = laneId / B_VEC_COLS;
    params.slabColIdxB = laneId % B_VEC_COLS;

    return params;
}