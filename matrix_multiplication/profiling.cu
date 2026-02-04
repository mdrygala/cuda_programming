#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

#include "gemm_config.h"
#include "kernels.cuh"



#define CHECK_CUDA(call) do {                                   \
  cudaError_t err = (call);                                     \
  if (err != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
            __FILE__, __LINE__, cudaGetErrorString(err));        \
    std::exit(1);                                               \
  }                                                             \
} while(0)

static void verifyGEMM_small(const float *A, const float *B,
                             const float *Cgpu, const float *Cinit,
                             int M, int N, int K, float alpha, float beta)
{
    const float atol = 1e-2f, rtol = 1e-2f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float ref = 0.0f;
            for (int k = 0; k < K; k++) ref += A[i*K + k] * B[k*N + j];
            ref = alpha * ref + beta * Cinit[i*N + j];

            float diff = std::fabs(Cgpu[i*N + j] - ref);
            float tol  = atol + rtol * std::fabs(ref);
            if (diff > tol) {
                fprintf(stderr,
                        "Mismatch (%d,%d): gpu=%f ref=%f diff=%f tol=%f\n",
                        i, j, Cgpu[i*N+j], ref, diff, tol);
                std::exit(2);
            }
        }
    }
}

int main() {

    
    // ---------- choose what to profile (change ONE line) ----------
    auto Kernel = GEMMSubtileFinal;   // GEMMBaseline, GEMMTiling, or GEMMSubtileFinal

    // ---------- basic config sanity ----------
    assert(SUBTILE % SUB == 0);

    // ---------- 1) VERIFY ON SMALL ----------
    {
        int M = 256, N = 256, K = 256;
        float alpha = 1.0f, beta = 1.0f;

        std::vector<float> A(M*K, 1.0f);
        std::vector<float> B(K*N, 1.0f);
        std::vector<float> C(M*N, 0.0f);
        std::vector<float> Cinit = C;

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CHECK_CUDA(cudaMalloc(&dA, sizeof(float)*M*K));
        CHECK_CUDA(cudaMalloc(&dB, sizeof(float)*K*N));
        CHECK_CUDA(cudaMalloc(&dC, sizeof(float)*M*N));

        CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dC, C.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));

        dim3 block, grid;

        // Launch dims:
        // - Baseline/Tiling often use TILE x TILE threads
        // - SubtileFinal uses (SUBTILE/SUB) x (SUBTILE/SUB) threads
        //
        // To keep this file simple, we pick:
        //   - if profiling subtile: use SUBTILE/SUB
        //   - else: use TILE (common)
        if (Kernel == GEMMSubtileFinal) {
            block = dim3(SUBTILE / SUB, SUBTILE / SUB, 1);
            grid  = dim3((N + SUBTILE - 1)/SUBTILE, (M + SUBTILE - 1)/SUBTILE, 1);
        } else {
            block = dim3(TILE, TILE, 1);
            grid  = dim3((N + TILE - 1)/TILE, (M + TILE - 1)/TILE, 1);
        }

        Kernel<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(C.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
        verifyGEMM_small(A.data(), B.data(), C.data(), Cinit.data(), M, N, K, alpha, beta);

        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));

        printf("Verification passed.\n");
    }

    // ---------- 2) PROFILE ON LARGE (ONLY THIS REGION) ----------
    {
        int N = 1 << 13;
        int M = N, K = N;
        float alpha = 1.0f, beta = 1.0f;

        std::vector<float> A(M*K, 1.0f);
        std::vector<float> B(K*N, 1.0f);
        std::vector<float> C(M*N, 0.0f);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CHECK_CUDA(cudaMalloc(&dA, sizeof(float)*M*K));
        CHECK_CUDA(cudaMalloc(&dB, sizeof(float)*K*N));
        CHECK_CUDA(cudaMalloc(&dC, sizeof(float)*M*N));

        CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, B.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dC, C.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice));

        dim3 block, grid;
        if (Kernel == GEMMSubtileFinal) {
            block = dim3(SUBTILE / SUB, SUBTILE / SUB, 1);
            grid  = dim3((N + SUBTILE - 1)/SUBTILE, (M + SUBTILE - 1)/SUBTILE, 1);
        } else {
            block = dim3(TILE, TILE, 1);
            grid  = dim3((N + TILE - 1)/TILE, (M + TILE - 1)/TILE, 1);
        }

        // warmup (not profiled)
        Kernel<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        const int iters = 10;

        CHECK_CUDA(cudaProfilerStart());
        for (int it = 0; it < iters; ++it) {
            Kernel<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
        }
        CHECK_CUDA(cudaProfilerStop());

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaFree(dA));
        CHECK_CUDA(cudaFree(dB));
        CHECK_CUDA(cudaFree(dC));

        printf("Profiling loop finished.\n");
    }

    return 0;
}
