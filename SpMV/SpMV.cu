#include <cuda_runtime.h>
#include "kernels.cuh"

void initialize(float* x, float* values, int* col_idx, int* row_ptr, int nnz){

}


int main(){

    int M = 1 << 20;
    int N = 1 << 20;
    int nnz = 1 << 15;

    float *dvalues, *hvalues, *dx, *dy, *hx, *hy;
    int *dcol_idx, *drow_ptr, *hcol_idx, *hrow_ptr;

    hvalues = new float[nnz];
    hx = new float[N];
    hcol_idx = new int[nnz];
    hrow_ptr = new int [M+1];
    hy = new float [M];

    cudaMalloc(&dx, sizeof(float) * N);
    cudaMalloc(&dy, sizeof(float) * M);
    cudaMalloc(&dvalues, sizeof(float) * nnz);
    cudaMalloc(&dcol_idx, sizeof(int) * nnz);
    cudaMalloc(&drow_ptr, sizeof(int) * (M+1));

    initialize(hx, hvalues, hcol_idx, hrow_ptr);

    cudaMemcpy(dx, hx, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dvalues, hvalues, sizeof(float) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(dcol_idx, hcol_idx, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(drow_ptr, hrow_ptr, sizeof(int) * (M+1), cudaMemcpyHostToDevice);


    int ThreadsPerBlock = 256;
    int WarpsPerBlock = ThreadsPerBlock >> 5;
    int BlocksPerGrid = (M + WarpsPerBlock - 1)/WarpsPerBlock;

    SpMVWarpRow<<<BlocksPerGrid, ThreadsPerBlock>>>(dvalues, dcol_idx, drow_ptr, dx, dy, M);

    cudaDeviceSynchronize();

    cudaMemcpy(hy, dy, sizeof(float) * M, cudaMemcpyDeviceToHost);

    delete[] hvalues;
    delete[] hx;
    delete[] hcol_idx;
    delete[] hrow_ptr;
    delete[] hy;

    cudaFree(dx);
    cudaFree(dvalues);
    cudaFree(dcol_idx);
    cudaFree(drow_ptr);
    cudaFree(dy);

    return 0;
}