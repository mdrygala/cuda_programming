#pragma once

__global__ void prefixSumExclusiveNaive(const int* __restrict__ input, const int* __restrict__ tree, int N);