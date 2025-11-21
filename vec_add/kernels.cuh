#pragma once

// used by Instruction-Level Parallelism (ILP) kernel (each thread handles UNROLL elems per step)
#ifndef UNROLL
#define UNROLL 4
#endif

// Vector addition with baseline kernel
__global__ void vecAddBaseline(const int* __restrict__ A,
                                const int* __restrict__ B,
                                int* __restrict__ C,
                                int N);

// Vector addition with Instruction-Level Parallelism (ILP)
__global__ void vecAddILP(const int* __restrict__ A,
                          const int* __restrict__ B,
                          int* __restrict__ C,
                          int N);

// Vector addition using int4 loads/stores (4 ints at once)
__global__ void vecAddInt4(const int* __restrict__ A,
                            const int* __restrict__ B,
                            int* __restrict__ C,
                            int N);
