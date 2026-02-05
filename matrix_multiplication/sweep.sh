#!/bin/bash

# Define the ranges
SUBTILE_SIZES=(32 64 128)
SUB_SIZES=(4 8 16)

echo "----------------------------------------------------------"
echo "STARTING PARAMETER SWEEP"
echo "----------------------------------------------------------"

for ST in "${SUBTILE_SIZES[@]}"; do
    for S in "${SUB_SIZES[@]}"; do
        # Logic check: alignment and block size
        if (( ST % S != 0 )) || (( S % 4 != 0 )); then continue; fi

        echo ">>> CONFIG: SUBTILE=$ST, SUB=$S"

        # Compile silently
        nvcc -O3 -DSUBTILE=$ST -DSUB=$S profiling.cu kernels.cu -o temp_bin 2>/dev/null
        
        if [ $? -ne 0 ]; then
            echo "    [!] Compilation Failed"
            continue
        fi

        # Run and print only the lines we care about
        # We use grep to show the performance and verification lines
        ./temp_bin | grep -iE "GFLOPS|TFLOPS|Time|Verification|Error" | sed 's/^/    /'

        echo "----------------------------------------------------------"
    done
done

rm -f temp_bin