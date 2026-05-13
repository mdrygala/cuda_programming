#!/bin/bash

TILE_MS=(32 64 128)
TILE_NS=(32 64 128)
TILE_KS=(8 16 32 64)
THREAD_DIMS=(4 8)
PADDINGS=(0 1 8)

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
echo "STARTING REGISTER VEC4 TRANSPOSED PARAMETER SWEEP"
echo "----------------------------------------------------------"

for TM in "${TILE_MS[@]}"; do
  for TN in "${TILE_NS[@]}"; do
    for TK in "${TILE_KS[@]}"; do
      for THREAD_DIM in "${THREAD_DIMS[@]}"; do
        for PAD in "${PADDINGS[@]}"; do

          if (( TM % THREAD_DIM != 0 )); then continue; fi
          if (( TN % THREAD_DIM != 0 )); then continue; fi

          THREADS_M=$(( TM / THREAD_DIM ))
          THREADS_N=$(( TN / THREAD_DIM ))
          THREADS=$(( THREADS_M * THREADS_N ))

          if (( THREADS <= 0 )); then continue; fi
          if (( THREADS > 1024 )); then continue; fi
          if (( THREADS % 32 != 0 )); then continue; fi

          CONFIG_STR="TILE_REGISTER_VEC_TRANSPOSED_M=$TM TILE_REGISTER_VEC_TRANSPOSED_N=$TN TILE_REGISTER_VEC_TRANSPOSED_K=$TK THREAD_DIM_REGISTER_VEC_TRANSPOSED=$THREAD_DIM PADDING_REGISTER_VEC_TRANSPOSED=$PAD THREADS=$THREADS"

          COMPILE_OUTPUT=$(nvcc -O3 -arch=sm_80 -lineinfo -Xptxas -v \
            -I. \
            -DTILE_REGISTER_VEC_TRANSPOSED_M=$TM \
            -DTILE_REGISTER_VEC_TRANSPOSED_N=$TN \
            -DTILE_REGISTER_VEC_TRANSPOSED_K=$TK \
            -DTHREAD_DIM_REGISTER_VEC_TRANSPOSED=$THREAD_DIM \
            -DPADDING_REGISTER_VEC_TRANSPOSED=$PAD \
            launch_subtiling_vectorized_transposed.cu \
            -o .build/register_vec_transposed_temp 2>&1)

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
    if (current_kernel ~ /^void GEMMSubTilingVec4Transposed/ || current_kernel ~ /^GEMMSubTilingVec4Transposed/) {
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
              OUTPUT=$(./.build/register_vec_transposed_temp 2>&1)
              RUN_LABEL="float"
            else
              OUTPUT=$(./.build/register_vec_transposed_temp --datatype half 2>&1)
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
      done
    done
  done
done

echo ""
echo "================ BEST REGISTER VEC4 TRANSPOSED CONFIGS ================"

for DT in "${DATATYPES[@]}"; do
  echo ""
  echo "DATATYPE:   $DT"
  echo "CONFIG:     ${BEST_CONFIG[$DT]}"
  echo "TFLOPS:     ${BEST_TFLOPS[$DT]}"
  echo "Efficiency: ${BEST_EFF[$DT]}%"
done

echo "======================================================================="

rm -f .build/register_vec_transposed_temp