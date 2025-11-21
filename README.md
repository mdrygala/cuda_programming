# CUDA Programming Experiments

This repository contains small CUDA experiments focused on understanding how
GPU kernels are written and how performance improves through incremental
optimization. At this stage the repository includes two core components:
a simple vector addition kernel and an increasingly optimized GEMM
implementation.

The goal of this work is to build intuition for CUDA programming, memory
hierarchy, and kernel performance.

---

## 1. Vector Addition

A minimal CUDA kernel used to learn:
- thread and block indexing
- mapping work to the GPU
- global memory access
- timing kernels with CUDA events

This serves as the baseline introduction to writing and launching device code.

---

## 2. GEMM (Matrix Multiplication)

A step by step exploration of matrix multiplication on the GPU. Each version
adds one optimization idea, making it possible to see how performance
improves.

Included kernels:
- baseline naive GEMM
- shared memory tiling
- register tiling
- register level sub tiling
- vectorized loads
- vectorized loads with shared memory swizzling

For each version we measure:
- kernel runtime
- achieved GFLOP/s
- percentage of theoretical FP32 peak

The goal is not to match cuBLAS, but to understand how each hardware feature
(concurrency, shared memory, registers, memory access patterns) contributes
to performance.

---