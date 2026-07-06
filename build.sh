#!/usr/bin/env bash
set -euo pipefail

# 编译 C++/CUDA 后端为 Python 扩展 my_project_backend
# 依赖: g++, pip install pybind11, fplll/gmp/mpfr(pkg-config), 可选 nvcc(CUDA)

SRC_DIR="src"
OUT_MODULE="my_project_backend$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

PYINC=$(python3.13 -m pybind11 --includes)
FPLLL_CFLAGS=$(pkg-config --cflags fplll)
FPLLL_LIBS=$(pkg-config --libs fplll)

CXX=${CXX:-g++}
COMMON="-O3 -std=c++17 -fPIC -funroll-loops -fomit-frame-pointer"

if command -v nvcc >/dev/null 2>&1; then
  echo ">> 检测到 CUDA, 编译 GPU 版本"
  CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}

  nvcc -O3 -std=c++17 -Xcompiler -fPIC \
    -c "$SRC_DIR/lattice_cuda.cu" -o lattice_cuda.o

  $CXX $COMMON -DUSE_CUDA $PYINC -I"$CUDA_HOME/include" $FPLLL_CFLAGS \
    -c "$SRC_DIR/lattice_backend.cpp" -o lattice_backend.o

  $CXX -shared lattice_backend.o lattice_cuda.o -o "$OUT_MODULE" \
    $FPLLL_LIBS -lgmp -lmpfr -L"$CUDA_HOME/lib64" -lcudart -lpthread
else
  echo ">> 未检测到 nvcc, 编译 CPU-only 版本"
  $CXX $COMMON $PYINC $FPLLL_CFLAGS \
    -c "$SRC_DIR/lattice_backend.cpp" -o lattice_backend.o
  $CXX -shared lattice_backend.o -o "$OUT_MODULE" \
    $FPLLL_LIBS -lgmp -lmpfr -lpthread
fi

rm -f lattice_backend.o lattice_cuda.o
echo ">> 已生成 $OUT_MODULE"
# 如需在任意目录 import, 复制到 site-packages:
# cp "$OUT_MODULE" "$(python3.13 -c 'import site;print(site.getsitepackages()[0])')/"
#bash build.sh 直接敲这行
