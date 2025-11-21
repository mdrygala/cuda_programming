#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build.sh <target_dir> [sm_arch] [ptxas_verbose]
#   target_dir:     e.g., vec_add  or  matrix_mult
#   sm_arch:        sm_80 (default), sm_75, sm_86, sm_90, ...
#   ptxas_verbose:  on (default) or off
#
# Examples:
#   ./build.sh vec_add
#   ./build.sh matrix_mult sm_86 off

TARGET="${1:-}"
ARCH="${2:-sm_80}"
PTXAS_VERBOSE="${3:-on}"

if [[ -z "${TARGET}" ]]; then
  echo "Usage: ./build.sh <target_dir> [sm_arch] [ptxas_verbose]"
  exit 1
fi

SRCDIR="${TARGET}"
OUTDIR="${SRCDIR}/.build"   # put .o and exe inside target/.build
BIN="${OUTDIR}/${TARGET}"

if [[ ! -d "${SRCDIR}" ]]; then
  echo "Error: directory '${SRCDIR}' not found."
  exit 1
fi

# Collect all .cu files in the target dir (single level)
mapfile -t SRCS < <(find "${SRCDIR}" -maxdepth 1 -type f -name '*.cu' | sort)
if [[ ${#SRCS[@]} -eq 0 ]]; then
  echo "Error: no .cu files found in '${SRCDIR}'."
  exit 1
fi

# Include paths: root include/ and per-target include/ if present
INC_FLAGS=""
[[ -d "include" ]] && INC_FLAGS+=" -Iinclude"
[[ -d "${SRCDIR}/include" ]] && INC_FLAGS+=" -I${SRCDIR}/include"

PTXAS_FLAGS=""
if [[ "${PTXAS_VERBOSE}" == "on" ]]; then
  PTXAS_FLAGS="-Xptxas -v"
fi

mkdir -p "${OUTDIR}"

# Compile each .cu into OUTDIR/*.o
OBJS=()
for SRC in "${SRCS[@]}"; do
  OBJ="${OUTDIR}/$(basename "${SRC}" .cu).o"
  echo "Compiling ${SRC} -> ${OBJ}"
  nvcc -O3 -std=c++17 -arch="${ARCH}" ${INC_FLAGS} ${PTXAS_FLAGS} \
       -rdc=true -c "${SRC}" -o "${OBJ}"
  OBJS+=("${OBJ}")
done

# Link all objects into a single executable
echo "Linking -> ${BIN}"
nvcc -arch="${ARCH}" "${OBJS[@]}" -o "${BIN}" -lcublas

echo "Running ${BIN} ..."
"./${BIN}"


# #!/usr/bin/env bash
# set -euo pipefail

# # Usage:
# #   ./build.sh [sm_arch] [ptxas_verbose]
# #   sm_arch:        sm_80 (default), sm_75, sm_86, sm_90, ...
# #   ptxas_verbose:  on (default) or off
# #
# # Example:
# #   ./build.sh sm_80 on
# #   ./build.sh sm_80 off

# ARCH="${1:-sm_80}"
# PTXAS_VERBOSE="${2:-on}"

# INC_FLAGS="-I../include"
# PTXAS_FLAGS=""
# if [[ "$PTXAS_VERBOSE" == "on" ]]; then
#   PTXAS_FLAGS="-Xptxas -v"
# fi

# mkdir -p build

# # Compile (need -rdc=true since we have multiple .cu TUs)
# nvcc -O3 -std=c++17 -arch="$ARCH" $INC_FLAGS $PTXAS_FLAGS -rdc=true -c kernels.cu -o build/kernels.o
# nvcc -O3 -std=c++17 -arch="$ARCH" $INC_FLAGS $PTXAS_FLAGS -rdc=true -c vec_add.cu -o build/vec_add.o

# # Link
# nvcc -arch="$ARCH" build/kernels.o build/vec_add.o -o vec_add

# echo "Running ./build/vec_add ..."
# ./build/vec_add