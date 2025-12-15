#include <cuda_runtime.h>
#include "kernels.cuh"
#include <vector>
#include <cstdio>
using namespace std;


int main(){

    int N = 1 << 13;
    int K = N;
    int M = N;

    float *hA, *hB, *hC;
    float *dA, *dB, *dC;

    vector<float> A = vector<float>(M*K, 1.0f);
    vector<float> B = vector<float>(K*N, 1.0f);
    vector<float> C = vector<float>(M*N, 0.0f);
    hA = A.data();
    hB = B.data();

    hC = C.data();

    cudaMalloc(&dA, sizeof(float)*M*K);
    cudaMalloc(&dB, sizeof(float)*K*N);
    cudaMalloc(&dC, sizeof(float)*M*N);
    cudaMemcpy(dA, hA, sizeof(float)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float)*K*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeof(float)*M*N, cudaMemcpyHostToDevice);

    int blockWidth = 32;
    int gridHeight = (M + blockWidth -1)/blockWidth;
    int gridWidth = (N + blockWidth -1)/blockWidth;

    dim3 gridSize(gridWidth, gridHeight);
    dim3 blockSize(blockWidth, blockWidth);

        // --- Query kernel attributes ---
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, GEMMTiling);

    printf("Kernel attributes:\n");
    printf("  Registers per thread: %d\n", attr.numRegs);
    printf("  Static shared memory per block: %zu bytes\n", attr.sharedSizeBytes);
    printf("  Constant memory: %zu bytes\n", attr.constSizeBytes);
    printf("  Local memory per thread: %zu bytes\n", attr.localSizeBytes);
        int maxBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        GEMMBaseline,
        blockWidth * blockWidth,
        0   // dynamic shared memory (change if you use it)
    );

    printf("Occupancy estimate:\n");
    printf("  Max active blocks per SM: %d\n", maxBlocksPerSM);



    GEMMTiling<<<gridSize, blockSize>>>(M, N, K, 1.0f, dA, dB, 1.0f, dC);

    cudaDeviceSynchronize();
    cudaMemcpy(hC, dC, sizeof(float)* M * N, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

}