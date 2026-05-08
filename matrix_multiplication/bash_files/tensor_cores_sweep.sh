#!/bin/bash

SUBTILE_MS=(64 128 256)
SUBTILE_NS=(64 128 256)
SUBTILE_KS=(16 32 64)

PADDINGS=(0 8 16)
WARP_MS=(1 2 4)
WARP_NS=(1 2 4)



KERNELS=(tensorcores tensorcoresDB)

FRAGMENT_M=16
FRAGMENT_N=16
FRAGMENT_K=16

mkdir -p .build

declare -A BEST_EFF
declare -A BEST_TFLOPS
declare -A BEST_CONFIG

for K in "${KERNELS[@]}"; do
  BEST_EFF[$K]=0
  BEST_TFLOPS[$K]=0
  BEST_CONFIG[$K]=""
done

echo "----------------------------------------------------------"
echo "STARTING TENSOR CORE PARAMETER SWEEP"
echo "----------------------------------------------------------"

for STM in "${SUBTILE_MS[@]}"; do
  for STN in "${SUBTILE_NS[@]}"; do

    if (( STN == 256 && STM == 256 )); then
      continue
    fi

    for STK in "${SUBTILE_KS[@]}"; do
      for PAD in "${PADDINGS[@]}"; do
        for WM in "${WARP_MS[@]}"; do
          for WN in "${WARP_NS[@]}"; do

            if (( STM % FRAGMENT_M != 0 )); then continue; fi
            if (( STN % FRAGMENT_N != 0 )); then continue; fi
            if (( STK % FRAGMENT_K != 0 )); then continue; fi

            WARP_TILE_M=$(( WM * FRAGMENT_M ))
            WARP_TILE_N=$(( WN * FRAGMENT_N ))

            if (( STM % WARP_TILE_M != 0 )); then continue; fi
            if (( STN % WARP_TILE_N != 0 )); then continue; fi

            NUM_WARPS_M=$(( STM / WARP_TILE_M ))
            NUM_WARPS_N=$(( STN / WARP_TILE_N ))
            NUM_WARPS=$(( NUM_WARPS_M * NUM_WARPS_N ))
            THREADS=$(( NUM_WARPS * 32 ))

            if (( THREADS <= 0 )); then continue; fi
            if (( THREADS > 1024 )); then continue; fi

            CONFIG_STR="STM=$STM STN=$STN STK=$STK PAD=$PAD WARP_M=$WM WARP_N=$WN THREADS=$THREADS"

            COMPILE_OUTPUT=$(nvcc -O3 -arch=sm_80 -lineinfo -Xptxas -v \
              -I. \
              -DSUBTILE_TENSOR_CORE_M=$STM \
              -DSUBTILE_TENSOR_CORE_N=$STN \
              -DSUBTILE_TENSOR_CORE_K=$STK \
              -DPADDING_TENSOR_CORE=$PAD \
              -DFRAGMENT_M=$FRAGMENT_M \
              -DFRAGMENT_N=$FRAGMENT_N \
              -DFRAGMENT_K=$FRAGMENT_K \
              -DWARP_M=$WM \
              -DWARP_N=$WN \
              profiling.cu kernels/*.cu \
              -o .build/temp_bin 2>&1)

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
    if (current_kernel ~ /^GEMMTensorCores(DB)?\(/) {
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
              echo "    [!] Compilation Failed"
              echo "----------------------------------------------------------"
              continue
            fi

            for KERNEL in "${KERNELS[@]}"; do
              OUTPUT=$(./.build/temp_bin --kernel "$KERNEL" --datatype half 2>&1)
              RUN_STATUS=$?

              echo "$OUTPUT" | grep -iE "GFLOPS|TFLOPS|Kernel time|Time|Verification|Mismatch|CUDA error|Efficiency|Achieved|Unknown|Aborted" | sed "s/^/    [$KERNEL] /"

              if [ $RUN_STATUS -ne 0 ]; then
                echo "    [!] Run failed: $KERNEL"
                continue
              fi

              TFLOPS=$(echo "$OUTPUT" | grep "Achieved:" | sed -E 's/.*Achieved:[[:space:]]*([0-9.]+) TFLOP.*/\1/')
              EFF=$(echo "$OUTPUT" | grep "Efficiency:" | sed -E 's/.*Efficiency:[[:space:]]*([0-9.]+)%.*/\1/')

              if [[ -n "$EFF" ]]; then
                IS_BEST=$(awk -v eff="$EFF" -v best="${BEST_EFF[$KERNEL]}" 'BEGIN { print (eff > best) ? 1 : 0 }')

                if (( IS_BEST == 1 )); then
                  BEST_EFF[$KERNEL]=$EFF
                  BEST_TFLOPS[$KERNEL]=$TFLOPS
                  BEST_CONFIG[$KERNEL]="kernel=$KERNEL $CONFIG_STR"
                fi
              fi
            done

            echo "----------------------------------------------------------"

          done
        done
      done
    done
  done
done

echo ""
echo "================ BEST CONFIGS ================"

for KERNEL in "${KERNELS[@]}"; do
  echo ""
  echo "KERNEL:     $KERNEL"
  echo "CONFIG:     ${BEST_CONFIG[$KERNEL]}"
  echo "TFLOPS:     ${BEST_TFLOPS[$KERNEL]}"
  echo "Efficiency: ${BEST_EFF[$KERNEL]}%"
done

echo "=============================================="

rm -f .build/temp_bin