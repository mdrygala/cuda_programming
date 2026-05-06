#!/bin/bash

SUBTILE_MNS=(64 128)
SUBTILE_KS=(16 32 64)
PADDINGS=(0 8 16)
WARP_MS=(1 2 4)
WARP_NS=(1 2 4)

KERNEL="tensorcores"

mkdir -p .build

BEST_EFF=0
BEST_TFLOPS=0
BEST_CONFIG=""

FRAGMENT_M=16
FRAGMENT_N=16
FRAGMENT_K=16

echo "----------------------------------------------------------"
echo "STARTING TENSOR CORE PARAMETER SWEEP"
echo "----------------------------------------------------------"

for STMN in "${SUBTILE_MNS[@]}"; do
  for STK in "${SUBTILE_KS[@]}"; do
    for PAD in "${PADDINGS[@]}"; do
      for WM in "${WARP_MS[@]}"; do
        for WN in "${WARP_NS[@]}"; do

          if (( STK % FRAGMENT_K != 0 )); then continue; fi

          WARP_TILE_M=$(( WM * FRAGMENT_M ))
          WARP_TILE_N=$(( WN * FRAGMENT_N ))

          if (( STMN % WARP_TILE_M != 0 )); then continue; fi
          if (( STMN % WARP_TILE_N != 0 )); then continue; fi

          NUM_WARPS_M=$(( STMN / WARP_TILE_M ))
          NUM_WARPS_N=$(( STMN / WARP_TILE_N ))
          NUM_WARPS=$(( NUM_WARPS_M * NUM_WARPS_N ))
          THREADS=$(( NUM_WARPS * 32 ))

          if (( THREADS <= 0 )); then continue; fi
          if (( THREADS > 1024 )); then continue; fi

          echo ">>> CONFIG: SUBTILE_MN=$STMN SUBTILE_K=$STK PAD=$PAD WARP_M=$WM WARP_N=$WN THREADS=$THREADS"

          COMPILE_OUTPUT=$(nvcc -O3 -arch=sm_80 -lineinfo -Xptxas -v \
            -I. \
            -DSUBTILE=$STMN \
            -DSUBTILE_MN=$STMN \
            -DSUBTILE_K=$STK \
            -DSUB=4 \
            -DPADDING=0 \
            -DPADDING_GEN_DIM=0 \
            -DPADDING_WARP=0 \
            -DPADDING_TENSOR_CORE=$PAD \
            -DFRAGMENT_M=$FRAGMENT_M \
            -DFRAGMENT_N=$FRAGMENT_N \
            -DFRAGMENT_K=$FRAGMENT_K \
            -DWARP_M=$WM \
            -DWARP_N=$WN \
            profiling.cu kernels/*.cu \
            -o .build/temp_bin 2>&1)

          COMPILE_STATUS=$?

          echo "$COMPILE_OUTPUT" | c++filt | awk '
          /Compiling entry function/ {
              name=$0;
              sub(/^.*Compiling entry function '\''/, "", name);
              sub(/'\'' for.*$/, "", name);
          }
          /Used/ {
              print "    ------------------------------------------------------------";
              print "    KERNEL: " name;
              print "    INFO:   " $0;
          }
          /error|Error|fatal|undefined reference/ {
              print "    " $0;
          }
          '

          if [ $COMPILE_STATUS -ne 0 ]; then
            echo "    [!] Compilation Failed"
            echo "----------------------------------------------------------"
            continue
          fi

          OUTPUT=$(./.build/temp_bin --kernel "$KERNEL" --datatype half 2>&1)

          echo "$OUTPUT" | grep -iE "GFLOPS|TFLOPS|Kernel time|Time|Verification|Mismatch|CUDA error|Efficiency|Achieved" | sed 's/^/    /'

          TFLOPS=$(echo "$OUTPUT" | grep "Achieved:" | sed -E 's/.*Achieved:[[:space:]]*([0-9.]+) TFLOP.*/\1/')
          EFF=$(echo "$OUTPUT" | grep "Efficiency:" | sed -E 's/.*Efficiency:[[:space:]]*([0-9.]+)%.*/\1/')

          if [[ -n "$EFF" ]]; then
            IS_BEST=$(awk -v eff="$EFF" -v best="$BEST_EFF" 'BEGIN { print (eff > best) ? 1 : 0 }')

            if (( IS_BEST == 1 )); then
              BEST_EFF=$EFF
              BEST_TFLOPS=$TFLOPS
              BEST_CONFIG="kernel=$KERNEL SUBTILE_MN=$STMN SUBTILE_K=$STK PAD=$PAD WARP_M=$WM WARP_N=$WN THREADS=$THREADS"
            fi
          fi

          echo "----------------------------------------------------------"
        done
      done
    done
  done
done

echo ""
echo "================ BEST CONFIG ================"
echo "$BEST_CONFIG"
echo "TFLOPS:     $BEST_TFLOPS"
echo "Efficiency: $BEST_EFF%"
echo "============================================="

rm -f .build/temp_bin