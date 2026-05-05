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
#define PADDING_GEN_DIM 1
#endif

#ifndef PADDING
#define PADDING 1
#endif

#define NUM_THREADS_X (SUBTILE_MN / SUB)
#define NUM_THREADS_Y (SUBTILE_MN / SUB)
#define NUM_THREADS_PER_BLOCK (NUM_THREADS_X * NUM_THREADS_Y)
#define NUM_WARPS_PER_BLOCK (NUM_THREADS_PER_BLOCK / 32)


#include <string>

struct Config{
    std::string kernel_type = "warpslab";

};