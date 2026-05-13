#!/bin/bash

set -u

mkdir -p .build

NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-sm_80}
COMMON_FLAGS="-O3 -arch=${ARCH} -lineinfo -Xptxas -v -I."

echo "=========================================================="
echo "RUNNING SELECTED GEMM KERNEL BENCHMARKS"
echo "=========================================================="
echo ""

compile_kernel() {
  local src="$1"
  local out="$2"
  shift 2
  local extra_defs=("$@")

  echo "----------------------------------------------------------"
  echo "Compiling: $src"
  echo "Output:    $out"

  if [ ${#extra_defs[@]} -gt 0 ]; then
    echo "Defines:   ${extra_defs[*]}"
  fi

  echo "----------------------------------------------------------"

  COMPILE_OUTPUT=$($NVCC $COMMON_FLAGS \
    "${extra_defs[@]}" \
    "$src" \
    -o "$out" 2>&1)

  COMPILE_STATUS=$?

  echo "$COMPILE_OUTPUT" | c++filt | grep -iE \
    "Compiling entry function|Used|error|Error|fatal|undefined reference|static assertion" || true

  if [ $COMPILE_STATUS -ne 0 ]; then
    echo ""
    echo "[!] Compilation failed for $src"
    echo "$COMPILE_OUTPUT"
    echo "----------------------------------------------------------"
    return 1
  fi

  echo "[+] Compilation succeeded"
  echo ""

  return 0
}

run_kernel() {
  local exe="$1"
  local datatype="$2"
  local label="$3"

  echo "----------------------------------------------------------"
  echo "Running:   $label"
  echo "Command:   $exe --datatype $datatype"
  echo "----------------------------------------------------------"

  OUTPUT=$("$exe" --datatype "$datatype" 2>&1)
  RUN_STATUS=$?

  echo "$OUTPUT" | grep -iE \
    "GFLOPS|TFLOPS|Kernel time|Time|Verification|Mismatch|CUDA error|Efficiency|Achieved|Unknown|Aborted" || true

  if [ $RUN_STATUS -ne 0 ]; then
    echo "[!] Run failed for $label"
    echo "$OUTPUT"
  else
    echo "[+] Run succeeded for $label"
  fi

  echo ""
}

echo "=========================================================="
echo "1. BASELINE"
echo "=========================================================="

if compile_kernel launch_baseline.cu .build/launch_baseline; then
  run_kernel .build/launch_baseline float "baseline / float"
  run_kernel .build/launch_baseline half  "baseline / half"
fi


echo "=========================================================="
echo "2. TILING"
echo "=========================================================="

if compile_kernel launch_tiling.cu .build/launch_tiling; then
  run_kernel .build/launch_tiling float "tiling / float"
  run_kernel .build/launch_tiling half  "tiling / half"
fi


echo "=========================================================="
echo "3. SUBTILING SCALAR / FLOAT"
echo "=========================================================="

if compile_kernel launch_subtiling_scalar.cu .build/launch_subtiling_scalar_float; then
  run_kernel .build/launch_subtiling_scalar_float float "subtiling scalar / float"
fi


echo "=========================================================="
echo "4. SUBTILING SCALAR / HALF / PADDING_REGISTER_SCALAR=1"
echo "=========================================================="

if compile_kernel launch_subtiling_scalar.cu .build/launch_subtiling_scalar_half_pad1 \
  -DPADDING_REGISTER_SCALAR=1; then
  run_kernel .build/launch_subtiling_scalar_half_pad1 half "subtiling scalar / half / pad=1"
fi


echo "=========================================================="
echo "5. SUBTILING VECTORIZED / FLOAT"
echo "=========================================================="

if compile_kernel launch_subtiling_vectorized.cu .build/launch_subtiling_vectorized_float; then
  run_kernel .build/launch_subtiling_vectorized_float float "subtiling vectorized / float"
fi


echo "=========================================================="
echo "6. SUBTILING VECTORIZED TRANSPOSED / FLOAT / PADDING=8"
echo "=========================================================="

if compile_kernel launch_subtiling_vectorized_transposed.cu .build/launch_subtiling_vectorized_transposed_float_pad8 \
  -DPADDING_REGISTER_VEC_TRANSPOSED=8; then
  run_kernel .build/launch_subtiling_vectorized_transposed_float_pad8 float "subtiling vectorized transposed / float / pad=8"
fi


echo "=========================================================="
echo "7. SUBTILING VECTORIZED TRANSPOSED / HALF"
echo "=========================================================="

if compile_kernel launch_subtiling_vectorized_transposed.cu .build/launch_subtiling_vectorized_transposed_half; then
  run_kernel .build/launch_subtiling_vectorized_transposed_half half "subtiling vectorized transposed / half"
fi


echo "=========================================================="
echo "8. SUBTILING WARP LOAD / FLOAT"
echo "=========================================================="

if compile_kernel launch_subtiling_warp_load.cu .build/launch_subtiling_warp_load_float; then
  run_kernel .build/launch_subtiling_warp_load_float float "subtiling warp load / float"
fi


echo "=========================================================="
echo "9. SUBTILING WARP LOAD / HALF / CUSTOM CONFIG"
echo "=========================================================="

if compile_kernel launch_subtiling_warp_load.cu .build/launch_subtiling_warp_load_half_custom \
  -DTILE_WARP_LOAD_N=128 \
  -DTILE_WARP_LOAD_K=16 \
  -DTHREAD_DIM_WARP_LOAD=8; then
  run_kernel .build/launch_subtiling_warp_load_half_custom half "subtiling warp load / half / custom"
fi


echo "=========================================================="
echo "10. LINEAR VECTORIZED LOAD / FLOAT"
echo "=========================================================="

if compile_kernel launch_subtiling_linear_vectorized_load.cu .build/launch_subtiling_linear_vectorized_load_float; then
  run_kernel .build/launch_subtiling_linear_vectorized_load_float float "subtiling linear vectorized load / float"
fi


echo "=========================================================="
echo "11. LINEAR VECTORIZED LOAD / HALF / CUSTOM CONFIG"
echo "=========================================================="

if compile_kernel launch_subtiling_linear_vectorized_load.cu .build/launch_subtiling_linear_vectorized_load_half_custom \
  -DTILE_LINEAR_LOAD_M=64 \
  -DTILE_LINEAR_LOAD_N=128 \
  -DTILE_LINEAR_LOAD_K=16 \
  -DTHREAD_DIM_LINEAR_LOAD=8; then
  run_kernel .build/launch_subtiling_linear_vectorized_load_half_custom half "subtiling linear vectorized load / half / custom"
fi