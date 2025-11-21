# CUDA GEMM

This repository implements and benchmarks different GPU kernel strategies for general matrix multiply (GEMM) on an NVIDIA A100-SXM4-80GB GPU (Ampere architecture, compute capability 8.0 / SM80). 


### GEMM Kernel Performance

| Kernel                                         | Time (ms) | GFLOP/s     |
|------------------------------------------------|-----------|-------------|
| Baseline                                       | 735.061   | 2991.62     |
| Basic Tiling (Shared Memory)                   | 538.724   | 4081.91     |
| Register Sub Tiling (Naive Loads)              | 231.419   | 9502.35     |
| Register Sub Tiling (Vectorized Loading)       | 208.751   | 10534.17    |
| Register Sub Tiling (Vec4 + Swizzle)           | 208.723   | 10535.61    |
| cuBLAS SGEMM (FP32 cores)                      | 115.467   | 19044.62    |

### Theoretical FP32 Peak



### Notes

Further optimizations (warp tiling, double buffering will be added soon), as well as a detailed explanation of the code and explanations of the performance results.
<!-- , cp.async, and tensor-core WMMA. -->
