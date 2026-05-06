#!/bin/bash

mkdir -p .build
rm -f .build/prof

COMPILE_OUTPUT=$(nvcc -O3 -arch=sm_80 -lineinfo -Xptxas -v \
  -I. \
  -DSUBTILE=64 -DSUBTILE_MN=64 -DSUBTILE_K=32 -DSUB=4 \
  -DPADDING_GEN_DIM=0 -DPADDING=0 -DPADDING_WARP=0 \
  profiling.cu kernels/*.cu \
  -o .build/prof 2>&1)

COMPILE_STATUS=$?

echo "$COMPILE_OUTPUT" | c++filt | awk '
/Compiling entry function/ {
    name=$0;
    sub(/^.*Compiling entry function '\''/, "", name);
    sub(/'\'' for.*$/, "", name);
    arch=$0;
}
/Used/ {
    print "------------------------------------------------------------";
    print "KERNEL: " name;
    print "ARCH:   " arch;
    print "INFO:   " $0;
}
/error|Error|fatal|undefined reference|multiple definition|No such file|undefined/ {
    print $0;
}
'

if [ $COMPILE_STATUS -ne 0 ]; then
    echo ""
    echo "Build failed."
    exit $COMPILE_STATUS
fi

echo ""
echo "Build succeeded: .build/prof"
echo ""

KERNELS=(
  baseline
  tiling
  registerscalar
  registerscalartransposed
  registervec4
  registervec4transposed
  warpslab
  warpslabgendim
  warpslabtransposed
  tensorcores
)

HALF_KERNELS=(
  baseline
  tiling
  registerscalar
  warpslab
  warpslabgendim
  tensorcores
)

run_kernel() {
    local kernel="$1"
    local datatype="$2"

    if [ -z "$datatype" ]; then
        echo ""
        echo ">>> --kernel $kernel"
        OUTPUT=$(./.build/prof --kernel "$kernel" 2>&1)
    else
        echo ""
        echo ">>> --kernel $kernel --datatype $datatype"
        OUTPUT=$(./.build/prof --kernel "$kernel" --datatype "$datatype" 2>&1)
    fi

    RUN_STATUS=$?

    # Print the full program output so roofline/debug lines are not hidden.
    echo "$OUTPUT" | sed 's/^/    /'

    if [ $RUN_STATUS -ne 0 ]; then
        if [ -z "$datatype" ]; then
            echo "    [!] Kernel run failed: $kernel"
        else
            echo "    [!] Kernel run failed: $kernel $datatype"
        fi
    fi

    echo "------------------------------------------------------------"
}

is_half_kernel() {
    local kernel="$1"

    for hk in "${HALF_KERNELS[@]}"; do
        if [ "$hk" = "$kernel" ]; then
            return 0
        fi
    done

    return 1
}

echo "------------------------------------------------------------"
echo "RUNNING KERNELS"
echo "------------------------------------------------------------"

for KERNEL in "${KERNELS[@]}"; do
    # tensorcores is half-only, so skip the float/default run.
    if [ "$KERNEL" != "tensorcores" ]; then
        run_kernel "$KERNEL" ""
    fi

    if is_half_kernel "$KERNEL"; then
        run_kernel "$KERNEL" "half"
    fi
done