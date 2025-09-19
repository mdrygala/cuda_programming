# CUDA Vector Addition 

This repository implements and benchmarks different GPU kernel strategies for vector addition on an NVIDIA A100-SXM4-80GB GPU (Ampere architecture, compute capability 8.0 / SM80).  
In this document we reason about GPU architecture, memory bandwidth and latency, kernel design, and performance optimization. We provide experiental results as well as an analysis.

---

## Overview

We compare three kernel implementations of `C = A + B`:

1. **Baseline Kernel** – straightforward element-wise addition.
2. **Instruction-Level Parallelism (ILP)** – loop with multiple additions per thread with unrolling to increase ILP per thread.
3. **Int4** – vectorized loads/stores (`Iint4`) to exploit wider memory transactions.

Each kernel is benchmarked with varying block sizes to analyze occupancy and throughput.

---

## Device Information

- **GPU**: NVIDIA A100-SXM4-80GB  
- **Number of Streaming Multiprocessors (SMs)**: 108  
- **Registers per SM**: 65,536  
- **Warp Size**: 32  
- **Max Threads per SM**: 2,048  
- **Max Warps per SM**: 64  
- **Theoretical Peak Bandwidth**: $\approx$ 2039 GB/s  

---

##  Benchmark Results


Measured on **NVIDIA A100-SXM4-80GB (Ampere, SM80)** with $N = 2^{27}$ elements.  
Reported times are best-of runs; bandwidth (BW) is an estimation effective global memory throughput.

| Threads / Block | Baseline Time (ms) | Baseline BW (GB/s) | ILP Time (ms) | ILP BW (GB/s) | Int4 Time (ms) | INT4 BW (GB/s) |
|-----------------|--------------------|--------------------|---------------|---------------|----------------|----------------|
| 128             | 0.987              | 1632.34            | 1.005         | 1602.31       | 0.924          | 1743.27        |
| 256             | 1.022              | 1576.07            | 1.025         | 1571.98       | 0.926          | 1739.23        |
| 512             | 1.010              | 1595.05            | 1.008         | 1598.13       | 0.920          | 1750.54        |
| 1024            | 0.993              | 1621.30            | 0.991         | 1624.44       | 0.912          | **1765.40**    |






| Kernel    | Registers / Thread |
|-----------|--------------------|
| Baseline  | 28                 |
| ILP       | 32                 |
| INT4      | 32                 |

---

**Observations:**  
- Baseline and ILP saturate around ~1600 GB/s ($\approx$ 80% of peak).  
- Int4 vectorization consistently outperforms, reaching 1765 GB/s ($\approx$ 87% of peak). 
- Baseline uses slightly fewer registers (28 vs. 32), but both give 100% theoretical occupancy.

##  Performance Analysis

CUDA programs can be limited by several common bottlenecks:

1. Amdahl’s Law (serial fraction)  
2. Communication overhead  
3. Synchronization  
4. Load imbalance  
5. **Memory bandwidth**  
6. **Global memory latency**  
7. Warp divergence  
8. Shared memory bank conflicts  
9. Network contention  

For **vector addition**, the computation is *embarrassingly parallel*: each element is updated independently, with no synchronization, no shared memory, and no divergence.  
This eliminates most bottlenecks, leaving only memory effects:

- **5. Memory Bandwidth:**  
  The kernel performs just one addition per two loads and one store, meaning that arithmetic intensity is extremely low. Hence performance is limited by how fast data can be moved to and from DRAM. This is why we measure GB/s and compare against theoretical peak.

- **6. Global Memory Latency:**  
  Accessing global memory takes hundreds of cycles.  
  CUDA hides this latency by running many warps concurrently.  Our kernel variants experiment with ways to better utilize memory bandwidth and reduce the impact of latency.  Our kernels achieve 100% theoretical occupancy.

**Result:** The optimized `Int4` kernel achieved $approx$ 87% of theoretical peak bandwidth showing that the design successfully addresses these bottlenecks.

---
## Repository Structure

- **vec_add.cu**  
  Entry point of the project. Handles:
  - Host setup (memory allocation, initialization, cleanup)  
  - Launching each kernel variant (Baseline, ILP, Int4)  
  - Collecting and printing benchmark results  

- **kernels.cu**  
  Device code with the actual CUDA kernels:  
  - `vecAddBaseline` – naive element-wise addition  
  - `vecAddILP` – unrolled loop for instruction-level parallelism  
  - `vecAddInt4` – vectorized addition using `int4` loads/stores  

- **kernels.cuh**  
  Header with kernel declarations, included in `vec_add.cu` for clean separation.  

- **include/benchmarking.hpp**  
  Benchmark utilities and device introspection:  
  - `GpuTimer` class (CUDA events for timing)  
  - `bench_best_ms` function (warmup, best-of iterations)  
  - Utility functions to read SM count, registers/SM, and peak memory bandwidth  
  - Prints device/SM capabilities  