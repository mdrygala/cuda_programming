# CUDA Programming Experiments

## Overview

This repository contains CUDA kernel implementations and experiments focused on understanding GPU performance and optimization. The work explores how different design choices—memory access patterns, parallelism strategies, and data movement—affect kernel efficiency.

The repository is intended as an exploratory and educational resource rather than a production library.

---

## Implemented Components

### 1. Vector Addition

A minimal CUDA kernel used to explore:

- thread and block indexing  
- mapping parallel work onto the GPU  
- global memory access patterns  
- kernel timing using CUDA events  

This serves as a baseline for understanding CUDA execution and performance measurement.

---

### 2. GEMM (Matrix Multiplication)

A step-by-step implementation of matrix multiplication on the GPU, where each version introduces a specific optimization strategy to improve performance.

#### Implemented optimizations:

- naive baseline GEMM  
- shared memory tiling  
- register tiling  
- register-level sub-tiling  
- vectorized memory loads  
- vectorized loads with shared memory swizzling  

#### Performance evaluation:

For each version, the following metrics are measured:

- kernel runtime  
- achieved GFLOP/s  
- percentage of theoretical FP32 peak performance  

The goal is not to match cuBLAS, but to understand how GPU architectural features—such as shared memory, registers, and memory access patterns—impact performance and scalability.
