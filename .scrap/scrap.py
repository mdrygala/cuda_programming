int readDeviceProperties(){
    printf("unroll value %d\n", UNROLL);
    int dev=0;
    cudaGetDevice(&dev);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    int sms   = prop.multiProcessorCount;     // SM count aim for 2–4 blocks/SM when using grid-stride.
    int warp  = prop.warpSize;                // usually 32
    int maxTB = prop.maxThreadsPerBlock;      // e.g., 1024
    int regPerSM = prop.regsPerMultiprocessor; 

    int l2 = 0; 
    cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev); // bytes

    // (Optional) theoretical memory bandwidth, rough:
    // For a quick “is my GB/s close to peak?” sense:
    double memClockKHz   = prop.memoryClockRate;   // kHz
    double busWidthBits  = prop.memoryBusWidth;    // bits
    double peakGBs = (memClockKHz * 1000.0) * (busWidthBits / 8.0) * 2 /*DDR*/ / 1e9;

    int minGrid=0, bestBlock=0;
    cudaOccupancyMaxPotentialBlockSize(&minGrid, &bestBlock, vecAdd, /*dynSmem*/0, /*limit*/0);
    // bestBlock is a good block size; grid can be max(minGrid, 4*sms

    printf("Num available SMs %d \n", sms);
    printf("Num resisters per SM %d \n", regPerSM );
    printf("memory clock rate %.3f\n", memClockKHz);
    printf("bus width bits %.3f\n", busWidthBits);
    printf("peak GBS %.3f\n", peakGBs);
    printf("memory clock rate %.3f\n", memClockKHz);
    printf("min grid: %d, best block for vector add: %d \n", minGrid, bestBlock);
    return sms;
}