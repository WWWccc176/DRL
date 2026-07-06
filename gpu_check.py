#!/usr/bin/env python3
import os
import sys

print("== 后端 CUDA ==")
os.environ.pop("LATTICE_DISABLE_CUDA", None)
try:
    sys.path.insert(0, "/home/pyjast1123/DRL/scripts/analyse")
    from backend_loader import import_backend
    be = import_backend()
    print("backend file:", be.__file__)
    print("cuda_available():", be.cuda_available())
except Exception as e:
    print("后端:", repr(e))

print("\n== G6K ==")
try:
    import g6k
    print("g6k 文件:", g6k.__file__)
    from g6k import SieverParams
    p = SieverParams()
    print("gpu 控制项:", [k for k in dir(p) if "gpu" in k.lower()])
except Exception as e:
    print("g6k:", repr(e))

print("\n== nvidia-smi ==")
os.system("nvidia-smi --query-gpu=name,memory.used --format=csv,noheader || true")
