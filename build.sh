#!/bin/bash
cd rustcore
cargo clean
maturin develop --release
cd ..
echo "Build Done! Ready to run Python."
#bash build.sh 直接敲这行
