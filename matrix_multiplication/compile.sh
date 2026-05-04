#!/bin/bash

mkdir -p .build
rm -f .build/prof

COMPILE_OUTPUT=$(nvcc -O3 -lineinfo -Xptxas -v \
  -I. \
  -DSUBTILE_MN=128 -DSUBTILE_K=16 -DSUB=8 \
  profiling.cu kernels/*.cu \
  -o .build/prof 2>&1)

COMPILE_STATUS=$?

echo "$COMPILE_OUTPUT" | c++filt | awk '
/Compiling entry function/ {
    name=$0;
    sub(/^.*Compiling entry function '\''/, "", name);
    sub(/'\'' for.*$/, "", name);
}
/Used/ {
    print "------------------------------------------------------------";
    print "KERNEL: " name;
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