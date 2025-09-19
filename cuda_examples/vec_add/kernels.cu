#include <cuda_runtime.h>
#include "kernels.cuh"

// Vector addition with baseline kernel
__global__ void vecAddBaseline(const int* __restrict__ A,
                                const int* __restrict__ B,
                                int* __restrict__ C,
                                int N){
    // Global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-wide stride
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }

}

// Vector addition with Instruction-Level Parallelism (ILP)
__global__ void vecAddILP(const int* __restrict__ A,
                          const int* __restrict__ B,
                          int* __restrict__ C,
                          int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += UNROLL * stride) {
        #pragma unroll
        // Instruction Level parallelism by unrolling loop 
        for (int k = 0; k < UNROLL; ++k) {
            int j = i + k * stride;
            if (j < N) C[j] = A[j] + B[j];
        }
    }
}

// Vector addition using int4 loads/stores (4 ints at once)
__global__ void vecAddInt4(const int* __restrict__ A,
                            const int* __restrict__ B,
                            int* __restrict__ C,
                            int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Number of full groups of 4 elements.
    int N4 = N >> 2;

    const int4* A4 = reinterpret_cast<const int4*>(A);
    const int4* B4 = reinterpret_cast<const int4*>(B);
    int4*       C4 = reinterpret_cast<int4*>(C);

    // Process 4 elements at a time
    for (int i = tid; i < N4; i += stride) {
        int4 a = A4[i];
        int4 b = B4[i];
        C4[i] = make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    /// Handle leftover elements if N is not divisible by 4
    int start = N4 << 2;            
    for (int i = start + tid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}