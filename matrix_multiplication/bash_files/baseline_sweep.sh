#!/bin/bash

BASELINE_TILES=(8 16 32)

mkdir -p .build

declare -A BEST_EFF
declare -A BEST_TFLOPS
declare -A BEST_CONFIG

DATATYPES=(float half)

for DT in "${DATATYPES[@]}"; do
  BEST_EFF[$DT]=0
  BEST_TFLOPS[$DT]=0
  BEST_CONFIG[$DT]=""
done

echo "----------------------------------------------------------"
echo "STARTING BASELINE PARAMETER SWEEP"
echo "----------------------------------------------------------"

for TILE in "${BASELINE_TILES[@]}"; do
  THREADS=$(( TILE * TILE ))

  if (( THREADS <= 0 )); then continue; fi
  if (( THREADS > 1024 )); then continue; fi

  CONFIG_STR="BASELINE_TILE=$TILE THREADS=$THREADS"

  COMPILE_OUTPUT=$(nvcc -O3 -arch=sm_80 -lineinfo -Xptxas -v \
    -I. \
    -DBASELINE_TILE=$TILE \
    launch_baseline.cu \
    -o .build/baseline_temp 2>&1)

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
    if (current_kernel ~ /^void GEMMBaseline/ || current_kernel ~ /^GEMMBaseline/) {
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

  for DT in "${DATATYPES[@]}"; do
    if [ "$DT" = "float" ]; then
      OUTPUT=$(./.build/baseline_temp 2>&1)
      RUN_LABEL="float"
    else
      OUTPUT=$(./.build/baseline_temp --datatype half 2>&1)
      RUN_LABEL="half"
    fi

    RUN_STATUS=$?

    echo "$OUTPUT" | grep -iE "GFLOPS|TFLOPS|Kernel time|Time|Verification|Mismatch|CUDA error|Efficiency|Achieved|Unknown|Aborted" | sed "s/^/    [$RUN_LABEL] /"

    if [ $RUN_STATUS -ne 0 ]; then
      echo "    [!] Run failed: $RUN_LABEL $CONFIG_STR"
      continue
    fi

    TFLOPS=$(echo "$OUTPUT" | grep "Achieved:" | sed -E 's/.*Achieved:[[:space:]]*([0-9.]+) TFLOP.*/\1/')
    EFF=$(echo "$OUTPUT" | grep "Efficiency:" | sed -E 's/.*Efficiency:[[:space:]]*([0-9.]+)%.*/\1/')

    if [[ -n "$EFF" ]]; then
      IS_BEST=$(awk -v eff="$EFF" -v best="${BEST_EFF[$DT]}" 'BEGIN { print (eff > best) ? 1 : 0 }')

      if (( IS_BEST == 1 )); then
        BEST_EFF[$DT]=$EFF
        BEST_TFLOPS[$DT]=$TFLOPS
        BEST_CONFIG[$DT]="datatype=$DT $CONFIG_STR"
      fi
    fi
  done

  echo "----------------------------------------------------------"
done

echo ""
echo "================ BEST BASELINE CONFIGS ================"

for DT in "${DATATYPES[@]}"; do
  echo ""
  echo "DATATYPE:   $DT"
  echo "CONFIG:     ${BEST_CONFIG[$DT]}"
  echo "TFLOPS:     ${BEST_TFLOPS[$DT]}"
  echo "Efficiency: ${BEST_EFF[$DT]}%"
done

echo "======================================================="

rm -f .build/baseline_temp