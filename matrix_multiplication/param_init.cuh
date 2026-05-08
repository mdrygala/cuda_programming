#pragma once
#include <cuda_runtime.h>


struct SlabParamsGenDim {
    int warpId;

    int slabRowIdxA;
    int slabColIdxA;

    int slabRowIdxB;
    int slabColIdxB;
};

template <typename InputT, int SUBDIM_M, int SUBDIM_N, int SUBDIM_K>
__device__ __forceinline__
SlabParamsGenDim make_linear_slab_params_gendim()
{ 
    SlabParamsGenDim params;


    int threadBlockIdx = threadIdx.y * blockDim.x + threadIdx.x;

    params.warpId  = threadBlockIdx >> 5;
    int laneId = threadBlockIdx & 31;

    

    constexpr int VEC_BYTES = 16;
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(InputT);
    constexpr int WARP_SIZE = 32;

    static_assert(VEC_BYTES % sizeof(InputT) == 0,
                  "InputT must divide 16 bytes");

    static_assert(SUBDIM_K % VEC_ELEMS == 0,
                  "SUBDIM_K must be divisible by VEC_ELEMS");

    static_assert(SUBDIM_N % VEC_ELEMS == 0,
                  "SUBDIM_N must be divisible by VEC_ELEMS");
   

    constexpr int A_VEC_COLS = SUBDIM_K  / VEC_ELEMS;
    constexpr int B_VEC_COLS = SUBDIM_N / VEC_ELEMS;

    static_assert(WARP_SIZE % A_VEC_COLS == 0,
                  "SUBDIM_K / VEC_ELEMS must divide 32");

    static_assert(WARP_SIZE % B_VEC_COLS == 0,
                  "SUBDIM_N / VEC_ELEMS must divide 32");

    static_assert(SUBDIM_M % (WARP_SIZE / A_VEC_COLS) == 0,
              "SUBDIM_M must be divisible by rows covered per warp for A");
    static_assert(SUBDIM_K % (WARP_SIZE / B_VEC_COLS) == 0,
              "SUBDIM_K must be divisible by rows covered per warp for B");

    params.slabRowIdxA = laneId / A_VEC_COLS;
    params.slabColIdxA = laneId % A_VEC_COLS;

    params.slabRowIdxB = laneId / B_VEC_COLS;
    params.slabColIdxB = laneId % B_VEC_COLS;

    return params;
}