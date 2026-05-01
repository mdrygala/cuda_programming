#pragma once
#include <cuda_runtime.h>


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
    // int newColTile;
};





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
    

    // int shared_segment = params.colTile >> 5;
    // int shared_bank_idx = params.colTile & 31;
    // int new_shared_bank_idx = (shared_segment + shared_bank_idx) & 31;
    // params.newColTile = (shared_segment << 5) + new_shared_bank_idx;

    // params.newColTile = params.colTile;

    // params.computeColTile = params.newColTile;

    return params;
}



template <typename Params>
__device__ __forceinline__
Params make_params();


template <>
__device__ __forceinline__
FakeSwizzleParams make_params<FakeSwizzleParams>()
{
    return make_fake_swizzle_params();
}
