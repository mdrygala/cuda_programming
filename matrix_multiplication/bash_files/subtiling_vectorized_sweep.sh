#!/bin/bash

TILES=(32 64 128)
THREAD_DIMS=(4 8)
PADDINGS=(0 1)

mkdir -p .build

BEST_EFF=0
BEST_TFLOPS=0
BEST_CONFIG=""

echo "----------------------------------------------------------"
echo "STARTING REGISTER VEC4 PARAMETER SWEEP"
echo "----------------------------------------------------------"

for TILE in "${TILES[@]}"; do
  for THREAD_DIM in "${THREAD_DIMS[@]}"; do
    for PAD in "${PADDINGS[@]}"; do

      if (( TILE % THREAD_DIM != 0 )); then continue; fi

      THREADS_PER_DIM=$(( TILE / THREAD_DIM ))
      THREADS=$(( THREADS_PER_DIM * THREADS_PER_DIM ))

      if (( THREADS <= 0 )); then continue; fi
      if (( THREADS > 1024 )); then continue; fi
      if (( THREADS % 32 != 0 )); then continue; fi

      CONFIG_STR="TILE_REGISTER_VEC=$TILE THREAD_DIM_REGISTER_VEC=$THREAD_DIM PADDING_REGISTER_VEC=$PAD THREADS=$THREADS"

      COMPILE_OUTPUT=$(nvcc -O3 -arch=sm_80 -lineinfo -Xptxas -v \
        -I. \
        -DTILE_REGISTER_VEC=$TILE \
        -DTHREAD_DIM_REGISTER_VEC=$THREAD_DIM \
        -DPADDING_REGISTER_VEC=$PAD \
        launch_subtiling_vectorized.cu kernels/subtiling_vectorized.cu \
        -o .build/vectorized_temp 2>&1)

      COMPILE_STATUS=$?

      echo "$COMPILE_OUTPUT" | c++filt | awk -v config="$CONFIG_STR" '
/Compiling entry function/ {
    current_kernel=$0;
    sub(/^.*Compiling entry function /, "", current_kernel);
    gsub(/\047/, "", current_kernel);
    sub(/ for.*$/, "", current_kernel);
    next;
}

/Used/ {
    if (current_kernel ~ /^GEMMSubTilingVec4/) {
        print "------------------------------------------------------------";
        print "CONFIG: " config;
        print "KERNEL: " current_kernel;
        print "INFO:   " $0;
    }
    next;
}

/error|Error|fatal|undefined reference|static assertion/ {
    print $0;
}
'

      if [ $COMPILE_STATUS -ne 0 ]; then
        echo "    [!] Compilation Failed for $CONFIG_STR"
        echo "----------------------------------------------------------"
        continue
      fi

      OUTPUT=$(./.build/vectorized_temp 2>&1)
      RUN_STATUS=$?

      echo "$OUTPUT" | grep -iE "GFLOPS|TFLOPS|Kernel time|Time|Verification|Mismatch|CUDA error|Efficiency|Achieved|Unknown|Aborted" | sed "s/^/    /"

      if [ $RUN_STATUS -ne 0 ]; then
        echo "    [!] Run failed: $CONFIG_STR"
        echo "----------------------------------------------------------"
        continue
      fi

      TFLOPS=$(echo "$OUTPUT" | grep "Achieved:" | sed -E 's/.*Achieved:[[:space:]]*([0-9.]+) TFLOP.*/\1/')
      EFF=$(echo "$OUTPUT" | grep "Efficiency:" | sed -E 's/.*Efficiency:[[:space:]]*([0-9.]+)%.*/\1/')

      if [[ -n "$EFF" ]]; then
        IS_BEST=$(awk -v eff="$EFF" -v best="$BEST_EFF" 'BEGIN { print (eff > best) ? 1 : 0 }')

        if (( IS_BEST == 1 )); then
          BEST_EFF=$EFF
          BEST_TFLOPS=$TFLOPS
          BEST_CONFIG="$CONFIG_STR"
        fi
      fi

      echo "----------------------------------------------------------"
    done
  done
done

echo ""
echo "================ BEST REGISTER VEC4 CONFIG ================"
echo "CONFIG:     $BEST_CONFIG"
echo "TFLOPS:     $BEST_TFLOPS"
echo "Efficiency: $BEST_EFF%"
echo "==========================================================="

rm -f .build/vectorized_temp