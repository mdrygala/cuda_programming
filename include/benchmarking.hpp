#pragma once
#include <cuda_runtime.h>



// For timing CUDA kernels
class GpuTimer {
public:
  GpuTimer() { 
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
 }
  ~GpuTimer() { 
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
 }

 // Record start of event
  void start(cudaStream_t s = 0) { 
    cudaEventRecord(start_, s); 
}
  // Record stop of event
  void stop (cudaStream_t s = 0) { 
    cudaEventRecord(stop_,  s);
}
// Return elapsed time between start and stop of event in ms
float ms() {
    cudaEventSynchronize(stop_);
    float t = 0.f;
    cudaEventElapsedTime(&t, start_, stop_);
    return t;
}

private:
  cudaEvent_t start_{}, stop_{};
};


// Run kernel multiple times and return best (min) runtime in ms
template<class LaunchFn>
float bench_best_ms(LaunchFn&& launch, int warmups = 1, int iters = 5, cudaStream_t st = 0) {
  // warm-up the GPU
  for (int w = 0; w < warmups; ++w){
    launch();
  } 
  cudaStreamSynchronize(st);

  float best = 1e30f;
  GpuTimer t;
  for (int i = 0; i < iters; ++i) {
    t.start(st);
    launch();           
    t.stop(st);
    float m = t.ms();
    if (m < best) best = m;
  }
  return best;
}

// Convert bytes moved + runtime to estimate effective bandwidth (GB/s)
double gbs(double bytes_moved, float ms) {
  return (bytes_moved / 1e9) / (ms / 1e3);
}

double gflops(double flops, float ms){ 
  return flops / 1e9 / (ms/1e3);
 }

// functions for computing device info
int readSMs() {
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  return prop.multiProcessorCount;
}

int readRegSm() {
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  return prop.regsPerMultiprocessor;
}

void print_sm_caps() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);

    int maxThreadsPerSM   = p.maxThreadsPerMultiProcessor;
    int warpSize          = p.warpSize;                 // typically 32
    int maxWarpsPerSM     = maxThreadsPerSM / warpSize; // e.g., 2048/32 = 64
    int maxBlocksPerSM    = p.maxBlocksPerMultiProcessor;

    std::printf("Device: %s\n", p.name);
    std::printf("warpSize: %d\n", warpSize);
    std::printf("maxThreadsPerSM: %d\n", maxThreadsPerSM);
    std::printf("maxWarpsPerSM: %d\n", maxWarpsPerSM);
    std::printf("maxBlocksPerSM: %d\n", maxBlocksPerSM);
}


double computePeakGBs(){
    int dev=0;
    cudaGetDevice(&dev);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    // (Optional) theoretical memory bandwidth, rough:
    // For a quick “is my GB/s close to peak?” sense:
    double memClockKHz   = prop.memoryClockRate;   // kHz
    double busWidthBits  = prop.memoryBusWidth;    // bits
    double peakGBs = (memClockKHz * 1000.0) * (busWidthBits / 8.0) * 2 /*DDR*/ / 1e9;

    return peakGBs;
}


static inline int fp32_cores_per_sm(int maj,int min){
    int cc = maj*10+min;
    switch(cc){
        case 30: case 32: case 35: case 37: return 192;
        case 50: case 52: case 53:          return 128;
        case 60:                             return 64;
        case 61:                             return 128;
        case 70:                             return 64;
        case 75:                             return 64;
        case 80:                             return 64;
        case 86:                             return 128;
        case 89:                             return 128;
        case 90:                             return 128;
        default:                             return 128;
    }
}

static inline double peakFP32_GFLOPs() {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, dev);

    const int sms  = p.multiProcessorCount;
    const int cores = fp32_cores_per_sm(p.major, p.minor);

    // p.clockRate is in kHz. For GFLOP/s:
    // GFLOPs = SMs * cores/SM * 2 (FMA) * clock_kHz / 1e6
    return (double)sms * (double)cores * 2.0 * (double)p.clockRate / 1e6;
}


