#!/bin/bash

set -u

mkdir -p .build
mkdir -p ncu_reports

NVCC=${NVCC:-nvcc}
NCU=${NCU:-ncu}
USE_SUDO_NCU=${USE_SUDO_NCU:-1}
ARCH=${ARCH:-sm_80}

# Profile only selected launches, not every kernel launch in your program.
# Example: skip first 10 launches, profile only the next 1 launch.
NCU_LAUNCH_SKIP=${NCU_LAUNCH_SKIP:-10}
NCU_LAUNCH_COUNT=${NCU_LAUNCH_COUNT:-1}

if [ "$USE_SUDO_NCU" -eq 1 ]; then
  NCU_CMD="sudo $NCU"
else
  NCU_CMD="$NCU"
fi

COMMON_FLAGS="-O3 -arch=${ARCH} -lineinfo -Xptxas -v -I."

NCU_FLAGS="--set full --force-overwrite --launch-skip ${NCU_LAUNCH_SKIP} --launch-count ${NCU_LAUNCH_COUNT}"

echo "=========================================================="
echo "GENERATING NSIGHT COMPUTE REPORTS FOR SELECTED GEMM KERNELS"
echo "=========================================================="
echo "NVCC:             $NVCC"
echo "NCU command:      $NCU_CMD"
echo "ARCH:             $ARCH"
echo "NCU launch skip:  $NCU_LAUNCH_SKIP"
echo "NCU launch count: $NCU_LAUNCH_COUNT"
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

report_base_from_source() {
  local src="$1"

  local base
  base=$(basename "$src" .cu)

  # Remove leading launch_
  base="${base#launch_}"

  echo "$base"
}

run_ncu_report() {
  local exe="$1"
  local datatype="$2"
  local src="$3"
  local variant_suffix="$4"
  local label="$5"

  local base
  base=$(report_base_from_source "$src")

  local report_name="${base}_${datatype}${variant_suffix}"
  local report_path="ncu_reports/${report_name}"

  echo "----------------------------------------------------------"
  echo "Profiling: $label"
  echo "Command:   $NCU_CMD $NCU_FLAGS -o $report_path $exe --datatype $datatype"
  echo "Report:    ${report_path}.ncu-rep"
  echo "----------------------------------------------------------"

  OUTPUT=$($NCU_CMD $NCU_FLAGS \
    -o "$report_path" \
    "$exe" --datatype "$datatype" 2>&1)

  RUN_STATUS=$?

  echo "$OUTPUT" | grep -iE \
    "GFLOPS|TFLOPS|Kernel time|Time|Verification|Mismatch|CUDA error|Efficiency|Achieved|Unknown|Aborted|==PROF==|Profiling|Report|ERR_NVGPUCTRPERM|launch-skip|launch-count" || true

  if [ $RUN_STATUS -ne 0 ]; then
    echo "[!] Nsight Compute run failed for $label"
    echo "$OUTPUT"
  else
    echo "[+] Nsight Compute report generated: ${report_path}.ncu-rep"
  fi

  echo ""
}

echo "=========================================================="
echo "1. BASELINE"
echo "=========================================================="

SRC=launch_baseline.cu
if compile_kernel "$SRC" .build/launch_baseline; then
  run_ncu_report .build/launch_baseline float "$SRC" "" "baseline / float"
  run_ncu_report .build/launch_baseline half  "$SRC" "" "baseline / half"
fi


echo "=========================================================="
echo "2. TILING"
echo "=========================================================="

SRC=launch_tiling.cu
if compile_kernel "$SRC" .build/launch_tiling; then
  run_ncu_report .build/launch_tiling float "$SRC" "" "tiling / float"
  run_ncu_report .build/launch_tiling half  "$SRC" "" "tiling / half"
fi


echo "=========================================================="
echo "3. SUBTILING SCALAR / FLOAT"
echo "=========================================================="

SRC=launch_subtiling_scalar.cu
if compile_kernel "$SRC" .build/launch_subtiling_scalar_float; then
  run_ncu_report .build/launch_subtiling_scalar_float float "$SRC" "" "subtiling scalar / float"
fi


echo "=========================================================="
echo "4. SUBTILING SCALAR / HALF / PADDING_REGISTER_SCALAR=1"
echo "=========================================================="

SRC=launch_subtiling_scalar.cu
if compile_kernel "$SRC" .build/launch_subtiling_scalar_half_pad1 \
  -DPADDING_REGISTER_SCALAR=1; then
  run_ncu_report .build/launch_subtiling_scalar_half_pad1 half "$SRC" "_pad1" "subtiling scalar / half / pad=1"
fi


echo "=========================================================="
echo "5. SUBTILING VECTORIZED / FLOAT"
echo "=========================================================="

SRC=launch_subtiling_vectorized.cu
if compile_kernel "$SRC" .build/launch_subtiling_vectorized_float; then
  run_ncu_report .build/launch_subtiling_vectorized_float float "$SRC" "" "subtiling vectorized / float"
fi


echo "=========================================================="
echo "6. SUBTILING VECTORIZED TRANSPOSED / FLOAT / PADDING=8"
echo "=========================================================="

SRC=launch_subtiling_vectorized_transposed.cu
if compile_kernel "$SRC" .build/launch_subtiling_vectorized_transposed_float_pad8 \
  -DPADDING_REGISTER_VEC_TRANSPOSED=8; then
  run_ncu_report .build/launch_subtiling_vectorized_transposed_float_pad8 float "$SRC" "_pad8" "subtiling vectorized transposed / float / pad=8"
fi


echo "=========================================================="
echo "7. SUBTILING VECTORIZED TRANSPOSED / HALF"
echo "=========================================================="

SRC=launch_subtiling_vectorized_transposed.cu
if compile_kernel "$SRC" .build/launch_subtiling_vectorized_transposed_half; then
  run_ncu_report .build/launch_subtiling_vectorized_transposed_half half "$SRC" "" "subtiling vectorized transposed / half"
fi


echo "=========================================================="
echo "8. SUBTILING WARP LOAD / FLOAT"
echo "=========================================================="

SRC=launch_subtiling_warp_load.cu
if compile_kernel "$SRC" .build/launch_subtiling_warp_load_float; then
  run_ncu_report .build/launch_subtiling_warp_load_float float "$SRC" "" "subtiling warp load / float"
fi


echo "=========================================================="
echo "9. SUBTILING WARP LOAD / HALF / CUSTOM CONFIG"
echo "=========================================================="

SRC=launch_subtiling_warp_load.cu
if compile_kernel "$SRC" .build/launch_subtiling_warp_load_half_custom \
  -DTILE_WARP_LOAD_N=128 \
  -DTILE_WARP_LOAD_K=16 \
  -DTHREAD_DIM_WARP_LOAD=8; then
  run_ncu_report .build/launch_subtiling_warp_load_half_custom half "$SRC" "_custom" "subtiling warp load / half / custom"
fi


echo "=========================================================="
echo "10. SUBTILING LINEAR VECTORIZED LOAD / FLOAT"
echo "=========================================================="

SRC=launch_subtiling_linear_vectorized_load.cu
if compile_kernel "$SRC" .build/launch_subtiling_linear_vectorized_load_float; then
  run_ncu_report .build/launch_subtiling_linear_vectorized_load_float float "$SRC" "" "subtiling linear vectorized load / float"
fi


echo "=========================================================="
echo "11. SUBTILING LINEAR VECTORIZED LOAD / HALF / CUSTOM CONFIG"
echo "=========================================================="

SRC=launch_subtiling_linear_vectorized_load.cu
if compile_kernel "$SRC" .build/launch_subtiling_linear_vectorized_load_half_custom \
  -DTILE_LINEAR_LOAD_M=64 \
  -DTILE_LINEAR_LOAD_N=128 \
  -DTILE_LINEAR_LOAD_K=16 \
  -DTHREAD_DIM_LINEAR_LOAD=8; then
  run_ncu_report .build/launch_subtiling_linear_vectorized_load_half_custom half "$SRC" "_custom" "subtiling linear vectorized load / half / custom"
fi


echo "=========================================================="
echo "DONE"
echo "Reports are in: ncu_reports/"
echo "=========================================================="