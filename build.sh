#!/bin/bash
cd rustcore
maturin develop --release
cd ..
echo "Build Done! Ready to run Python."
#./build.sh 直接敲这行
