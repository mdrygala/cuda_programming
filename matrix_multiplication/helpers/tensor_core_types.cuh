#pragma once

#include <cuda_fp16.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

template <int FRAG_DIM_M, int FRAG_DIM_N, int FRAG_DIM_K>
using AFrag = wmma::fragment<
    wmma::matrix_a,
    FRAG_DIM_M,
    FRAG_DIM_N,
    FRAG_DIM_K,
    __half,
    wmma::row_major
>;

template <int FRAG_DIM_M, int FRAG_DIM_N, int FRAG_DIM_K>
using BFrag = wmma::fragment<
    wmma::matrix_b,
    FRAG_DIM_M,
    FRAG_DIM_N,
    FRAG_DIM_K,
    __half,
    wmma::row_major
>;

template <int FRAG_DIM_M, int FRAG_DIM_N, int FRAG_DIM_K>
using AccFrag = wmma::fragment<
    wmma::accumulator,
    FRAG_DIM_M,
    FRAG_DIM_N,
    FRAG_DIM_K,
    float
>;