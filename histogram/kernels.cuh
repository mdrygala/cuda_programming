#pragma once

__global__ void histogramBaseline(const int* __restrict__ x, int* bins, int N);

__global__ void histogramBlockShare(const int* __restrict__ x, int* bins, int N, int B);