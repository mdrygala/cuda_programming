#pragma once
#ifndef TILE
#define TILE 32
#endif

#ifndef SUBTILE
#define SUBTILE 64
#endif

#ifndef SUB
#define SUB 4
#endif

#ifndef SWZ_MASK
#define SWZ_MASK 1      // try 1, then 3 if needed
#endif

#ifndef SWZ_SHIFT
#define SWZ_SHIFT 2     // XOR by 4 columns (float4-friendly)
#endif

#include <string>

struct Config{
    std::string kernel_type = "warp slab";

};