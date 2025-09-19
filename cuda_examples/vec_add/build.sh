#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build.sh [sm_arch] [ptxas_verbose]
#   sm_arch:        sm_80 (default), sm_75, sm_86, sm_90, ...
#   ptxas_verbose:  on (default) or off
#
# Example:
#   ./build.sh sm_80 on
#   ./build.sh sm_80 off

ARCH="${1:-sm_80}"
PTXAS_VERBOSE="${2:-on}"

INC_FLAGS="-I../include"
PTXAS_FLAGS=""
if [[ "$PTXAS_VERBOSE" == "on" ]]; then
  PTXAS_FLAGS="-Xptxas -v"
fi

mkdir -p build

# Compile (need -rdc=true since we have multiple .cu TUs)
nvcc -O3 -std=c++17 -arch="$ARCH" $INC_FLAGS $PTXAS_FLAGS -rdc=true -c kernels.cu -o build/kernels.o
nvcc -O3 -std=c++17 -arch="$ARCH" $INC_FLAGS $PTXAS_FLAGS -rdc=true -c vec_add.cu -o build/vec_add.o

# Link
nvcc -arch="$ARCH" build/kernels.o build/vec_add.o -o vec_add

echo "Running ./build/vec_add ..."
./build/vec_add