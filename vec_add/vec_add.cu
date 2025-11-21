//Computes the sum of two vectors on the GPU using CUDA.

#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include "kernels.cuh"
#include "../include/benchmarking.hpp"

// Fill two vectors A and B with random ints in [0, 100)
void initializeVectors(int *A, int *B, int N){
    for(int i = 0; i < N; i++){
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }
}

// Verify the vector add computation on CPU
void verify(int *A, int *B, int *C, int N){
    for(int i = 0; i < N; i++){
        assert(C[i] == A[i] + B[i]);
    }
}


int main(){
    // Print device informationn
    int sms = readSMs();
    double peakGBs = computePeakGBs();
    printf("peak GBS %.3f\n", peakGBs);
    int RegSM = readRegSm();
    printf("SMs %d and Registers per SM %d\n", sms, RegSM);
    print_sm_caps();

    // Set our problem size
    int N = 1<<27;
    size_t bytes = N * sizeof(int);
    size_t totalBytes = 3.0 * bytes;

    //Allocate unified memory for vectors
    int *A, *B, *C;

    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // Initialize input vectors
    initializeVectors(A, B, N);

    int block_mult = 8;
    
    // run Baseline kernel with different block sizes
    for (int ThreadsPerBlock : {128, 256, 512, 1024}){

        int BlocksPerGrid = std::min((N + ThreadsPerBlock - 1) / ThreadsPerBlock, block_mult * sms);
        float ms = bench_best_ms([&]{ vecAddBaseline<<<BlocksPerGrid,ThreadsPerBlock>>>(A,B,C,N);});
        printf("Kernel time for %d threads with Baseline : %.3f ms  |  BW: %.2f GB/s\n",
             ThreadsPerBlock, ms, gbs(totalBytes, ms));
        
        

    }
    // run ILP kernel with different block sizes
    for (int ThreadsPerBlock : {128, 256, 512, 1024}){
         
        int BlocksPerGrid = std::min((N + ThreadsPerBlock - 1) / ThreadsPerBlock, block_mult * sms);
        float ms = bench_best_ms([&]{ vecAddILP<<<BlocksPerGrid,ThreadsPerBlock>>>(A,B,C,N);});
        printf("Kernel time for %d threads with ILP : %.3f ms  |  BW: %.2f GB/s\n",
             ThreadsPerBlock, ms, gbs(totalBytes, ms));
        



    }
    // run Int4 kernel with different block sizes
    for (int ThreadsPerBlock : {128, 256, 512, 1024}){
        int BlocksPerGrid = std::min((N + ThreadsPerBlock - 1) / ThreadsPerBlock, block_mult * sms);
        float ms = bench_best_ms([&]{ vecAddInt4<<<BlocksPerGrid,ThreadsPerBlock>>>(A,B,C,N);});
        printf("Kernel time for %d threads with INT4 : %.3f ms  |  BW: %.2f GB/s\n",
             ThreadsPerBlock, ms, gbs(totalBytes, ms));
    }

   

    //Free GPU Memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);


    return 0;
}
