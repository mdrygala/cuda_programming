#include <cuda_runtime.h>
#include "kernels.cuh"
#include <algorithm>

void initialize(int* x, int* bins, int N, int B) {
    for (int i = 0; i < N; i++) {
        x[i] = i % B;    // example distribution
    }
    for (int i = 0; i < B; i++) {
        bins[i] = 0;
    }
}

int main(){

    int B = 1 << 10;
    int N = 1 << 23;

    int *bins, *x;

    cudaMallocManaged(&bins, B * sizeof(int));//allocate memory on GPU and CPU
    cudaMallocManaged(&x, N * sizeof(int));


    initialize(x, bins, N, B);

    int ThreadsPerBlock = 256;

    int numBlocks = std::min((N+ThreadsPerBlock-1)/ThreadsPerBlock, 1024);

    int numWarpsPerBlock = ThreadsPerBlock >> 5;

    histogramWarpPrivate<<<numBlocks, ThreadsPerBlock, B * sizeof(int) * numWarpsPerBlock>>>(x, bins, N, B);
    cudaDeviceSynchronize();

    cudaFree(x);
    cudaFree(bins);


        


    return 0;
}