#!/bin/bash

SUBTILES=(32 64 128)
SUBS=(4 8)

KERNEL="warpslab"

mkdir -p .build

BEST_EFF=0
BEST_TFLOPS=0
BEST_CONFIG=""

echo "----------------------------------------------------------"
echo "STARTING WARPSLAB PARAMETER SWEEP"
echo "----------------------------------------------------------"

for SUB in "${SUBS[@]}"; do
  for ST in "${SUBTILES[@]}"; do

    if (( ST % SUB != 0 )); then continue; fi

    THREADS=$(( (ST / SUB) * (ST / SUB) ))

    if (( THREADS % 32 != 0 )); then continue; fi
    if (( THREADS > 1024 )); then continue; fi

    echo ">>> CONFIG: SUBTILE=$ST SUB=$SUB THREADS=$THREADS"

    COMPILE_OUTPUT=$(nvcc -O3 -arch=sm_80 -lineinfo -Xptxas -v \
      -I. \
      -DSUBTILE=$ST \
      -DSUBTILE_MN=$ST \
      -DSUBTILE_K=$ST \
      -DSUB=$SUB \
      -DPADDING=0 \
      -DPADDING_GEN_DIM=0 \
      -DPADDING_WARP=0 \
      -DPADDING_TENSOR_CORE=0 \
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

    OUTPUT=$(./.build/temp_bin --kernel "$KERNEL" 2>&1)

    echo "$OUTPUT" | grep -iE "GFLOPS|TFLOPS|Time|Verification|Mismatch|CUDA error|Efficiency|Achieved" | sed 's/^/    /'

    TFLOPS=$(echo "$OUTPUT" | grep "Achieved:" | sed -E 's/.*\(([0-9.]+) TFLOPS\).*/\1/')
    EFF=$(echo "$OUTPUT" | grep "Efficiency:" | sed -E 's/.*Efficiency:[[:space:]]*([0-9.]+)%.*/\1/')

    if [[ -n "$EFF" ]]; then
      IS_BEST=$(awk -v eff="$EFF" -v best="$BEST_EFF" 'BEGIN { print (eff > best) ? 1 : 0 }')

      if (( IS_BEST == 1 )); then
        BEST_EFF=$EFF
        BEST_TFLOPS=$TFLOPS
        BEST_CONFIG="kernel=$KERNEL SUBTILE=$ST SUB=$SUB THREADS=$THREADS"
      fi
    fi

    echo "----------------------------------------------------------"
  done
done

echo ""
echo "================ BEST CONFIG ================"
echo "$BEST_CONFIG"
echo "TFLOPS:     $BEST_TFLOPS"
echo "Efficiency: $BEST_EFF%"
echo "============================================="

rm -f .build/temp_bin