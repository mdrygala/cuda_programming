#pragma once
#include <cuda_runtime.h>

struct NaiveParams{
    int threadRowTile;
    int threadColTile;
    int computeColTile;
};

struct SlabParams {
    int threadRowTile;
    int threadColTile;
    int computeColTile;

    int slabDimRows;
    int slabDimCols;
    int totalSlabs;

    int numWarps;
    int warpId;
    int slabRowIdx;
    int slabColIdx;
};

struct SwizzleParams {
    int threadRowTile;
    int threadColTile;
    int computeColTile;

    int slabDimRows;
    int slabDimCols;
    int numWarps;

    int warpRowGroup;
    int warpsPerColGroup;
    int slabRowIdx;

    int colTile;
    int newColTile;
};

struct FakeSwizzleParams {
    int threadRowTile;
    int threadColTile;
    int computeColTile;

    int slabDimRows;
    int slabDimCols;
    int numWarps;

    int warpRowGroup;
    int warpsPerColGroup;
    int slabRowIdx;

    int colTile;
    int newColTile;
};


__device__ __forceinline__
NaiveParams make_naive_params()
{
    NaiveParams params;
    params.threadRowTile = threadIdx.y * SUB;
    params.threadColTile = threadIdx.x * SUB;
    params.computeColTile = params.threadColTile;
    return params;
}

__device__ __forceinline__
SlabParams make_slab_params()
{
    SlabParams params;

    params.threadRowTile = threadIdx.y * SUB;
    params.threadColTile = threadIdx.x * SUB;
    params.computeColTile = params.threadColTile;

    params.slabDimRows = (SUBTILE + 3) >> 2;
    params.slabDimCols = (SUBTILE + 31) >> 5;
    params.totalSlabs  = params.slabDimRows * params.slabDimCols;

    params.numWarps = (blockDim.x * blockDim.y) >> 5;

    int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;
    params.warpId = threadBlockIdx >> 5;

    int laneId = threadBlockIdx & 31;
    params.slabRowIdx = laneId >> 3;
    params.slabColIdx = laneId & 7;

    return params;
}

__device__ __forceinline__
SwizzleParams make_swizzle_params()
{
    SwizzleParams params;

    params.threadRowTile = threadIdx.y * SUB;
    params.threadColTile = threadIdx.x * SUB;
    

    params.slabDimRows = (SUBTILE + 3) >> 2;
    params.slabDimCols = (SUBTILE + 31) >> 5;
    params.numWarps    = (blockDim.x * blockDim.y) >> 5;

    int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = threadBlockIdx >> 5;
    int laneId = threadBlockIdx & 31;

    int warpColGroup = warpId % params.slabDimCols;
    params.warpRowGroup = warpId / params.slabDimCols;
    params.warpsPerColGroup = params.numWarps / params.slabDimCols;

    params.slabRowIdx = laneId >> 3;
    int slabColIdx = laneId & 7;

    params.colTile = 32 * warpColGroup + 4 * slabColIdx;
    

    int shared_segment = params.colTile >> 5;
    int shared_bank_idx = params.colTile & 31;
    int new_shared_bank_idx = (shared_segment + shared_bank_idx) & 31;
    params.newColTile = (shared_segment << 5) + new_shared_bank_idx;

    params.computeColTile = params.newColTile;

    return params;
}


__device__ __forceinline__
FakeSwizzleParams make_fake_swizzle_params()
{
    FakeSwizzleParams params;

    params.threadRowTile = threadIdx.y * SUB;
    params.threadColTile = threadIdx.x * SUB;
    

    params.slabDimRows = (SUBTILE + 3) >> 2;
    params.slabDimCols = (SUBTILE + 31) >> 5;
    params.numWarps    = (blockDim.x * blockDim.y) >> 5;

    int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = threadBlockIdx >> 5;
    int laneId = threadBlockIdx & 31;

    int warpColGroup = warpId % params.slabDimCols;
    params.warpRowGroup = warpId / params.slabDimCols;
    params.warpsPerColGroup = params.numWarps / params.slabDimCols;

    params.slabRowIdx = laneId >> 3;
    int slabColIdx = laneId & 7;

    params.colTile = 32 * warpColGroup + 4 * slabColIdx;
    

    int shared_segment = params.colTile >> 5;
    int shared_bank_idx = params.colTile & 31;
    int new_shared_bank_idx = (shared_segment + shared_bank_idx) & 31;
    params.newColTile = (shared_segment << 5) + new_shared_bank_idx;

    params.newColTile = params.colTile;

    params.computeColTile = params.newColTile;

    return params;
}



template <typename Params>
__device__ __forceinline__
Params make_params();

template <>
__device__ __forceinline__
NaiveParams make_params<NaiveParams>()
{
    return make_naive_params();
}

template <>
__device__ __forceinline__
SlabParams make_params<SlabParams>()
{
    return make_slab_params();
}

template <>
__device__ __forceinline__
SwizzleParams make_params<SwizzleParams>()
{
    return make_swizzle_params();
}


template <>
__device__ __forceinline__
FakeSwizzleParams make_params<FakeSwizzleParams>()
{
    return make_fake_swizzle_params();
}
