#pragma once
#ifndef TILE
#define TILE 32
#endif

#ifndef SUBTILE
#define SUBTILE 64
#endif

#ifndef SUBTILE_MN
#define SUBTILE_MN 64
#endif


#ifndef SUBTILE_K
#define SUBTILE_K 64
#endif

#ifndef SUB
#define SUB 4
#endif

#ifndef PADDING_GEN_DIM
#define PADDING_GEN_DIM 0
#endif

#ifndef PADDING_WARP
#define PADDING_WARP 0
#endif

#ifndef PADDING
#define PADDING 0
#endif

#define NUM_THREADS_X (SUBTILE_MN / SUB)
#define NUM_THREADS_Y (SUBTILE_MN / SUB)
#define NUM_THREADS_PER_BLOCK (NUM_THREADS_X * NUM_THREADS_Y)
#define NUM_WARPS_PER_BLOCK (NUM_THREADS_PER_BLOCK / 32)



//Tensor Core:
#ifndef PADDING_TENSOR_CORE
#define PADDING_TENSOR_CORE 16
#endif

#ifndef FRAGMENT_M
#define FRAGMENT_M 16
#endif

#ifndef FRAGMENT_N
#define FRAGMENT_N 16
#endif

#ifndef FRAGMENT_K
#define FRAGMENT_K 16
#endif

#ifndef WARP_M  
#define WARP_M 2
#endif

#ifndef WARP_N  
#define WARP_N 2
#endif

#define WARP_TILE_M (WARP_M * FRAGMENT_M)
#define WARP_TILE_N (WARP_N * FRAGMENT_N)
#define NUM_WARPS_M (SUBTILE_MN / WARP_TILE_M)
#define NUM_WARPS_N (SUBTILE_MN / WARP_TILE_N)
#define NUM_WARPS_PER_BLOCK_TENSOR_CORE (NUM_WARPS_M *  NUM_WARPS_N)
#define NUM_THREADS_PER_BLOCK_TENSOR_CORE (NUM_WARPS_PER_BLOCK_TENSOR_CORE  * 32)


#include <string>

enum class DataType {
    Float32,
    Float16
};


struct Config{
    std::string kernel_type = "warpslab";
    DataType data_type = DataType::Float32;
};