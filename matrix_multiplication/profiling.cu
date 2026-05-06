#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <iostream>

#include "config.h"
#include "profiling.cuh"





void parseArgs(int argc, char** argv, Config& config);


int main(int argc, char** argv) {
    static_assert(SUBTILE % (SUB * SUB) == 0, "SUB^2 must divide SUBTILE");
    Config config;
    parseArgs(argc, argv, config);
    
    

    // ---------- basic config sanity ----------
    assert(SUBTILE % SUB == 0);


    if (config.data_type == DataType::Float32){
        // ---------- 1) VERIFY ON SMALL ----------
    
        run_verification<float>(256, 256, 256, config);
   
        // ---------- 2) PROFILE ON LARGE (ONLY THIS REGION) ----------
        run_profile<float>(1 << 12, 1 << 12, 1 << 12, config);
    }
    else if (config.data_type == DataType::Float16){
        // ---------- 1) VERIFY ON SMALL ----------
        int M, N, K;
        M = 256;
        N = M;
        K = M;
        if (config.kernel_type == "tensorcores"){
            if (M % FRAGMENT_M != 0 ||
                N % FRAGMENT_N != 0 ||
                K % FRAGMENT_K != 0) {
                throw std::runtime_error(
                    "GEMMTensorCores requires M, N, K to be multiples of 16"
                );
            }
        }
        run_verification<__half>(M, N, K, config);
   
        // ---------- 2) PROFILE ON LARGE (ONLY THIS REGION) ----------
        M = 1 << 12;
        N = M;
        K = M;
        run_profile<__half>(M, N, K, config);
    }
    else{
        throw std::runtime_error("Only allow for datatype of float or half");
    }
    

    return 0;
}


void parseArgs(int argc, char** argv, Config& config){
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--kernel" && i + 1 < argc){
            config.kernel_type = argv[++i];

        }
        else if(arg == "--datatype"  && i + 1 < argc){
            std::string datatype = argv[++i];

            if (datatype == "float" || datatype == "fp32") {
                config.data_type = DataType::Float32;
            }
            else if (datatype == "half" || datatype == "fp16") {
                config.data_type = DataType::Float16;
            }
            else {
                throw std::runtime_error("Unknown --datatype. Use float/fp32 or half/fp16.");
            }
        }
}
}

