#include <cuda_runtime.h>
#include "kernels.cuh"


int main()
{

    int N = 1 << 27;

    float *dinput, *dblockResults;
    size_t vecSize = N * sizeof(float);
    int numThreadsPerBlock = 256;
    int numBlocks = std::min((N + numThreadsPerBlock - 1) / numThreadsPerBlock, 1024);

    cudaMalloc(&dinput, vecSize);

    std::vector<float> hinput(N, 1.0f);
    cudaMemcpy(dinput, hinput.data(), vecSize, cudaMemcpyHostToDevice);

    cudaMalloc(&dblockResults, numBlocks * sizeof(float));

    float *inputPtr = dinput;
    float *blockResultsptr = dblockResults;


    while(true){
        SumReductionBaseline<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock*sizeof(float)>>>(inputPtr, blockResultsptr, N);
        cudaDeviceSynchronize();
        if (numBlocks == 1) break;

        N = numBlocks;
        numBlocks = std::min((N + numThreadsPerBlock - 1) / numThreadsPerBlock, 1024);
        std::swap(inputPtr, blockResultsptr);

        
    }

    float totalSum;
    CudaMemcpy(&totalSum, blockResultsptr, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Total Sum: " << totalSum << std::endl;

    cudaFree(dinput);
    cudaFree(dblockResults);
 
    return 0;
}