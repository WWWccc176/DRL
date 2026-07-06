#!/usr/bin/env python3
"""Check what will actually use the RTX 5080."""

import os

print("== backend CUDA ==")
os.environ.pop("LATTICE_DISABLE_CUDA", None)
try:
    from backend_loader import import_backend

    be = import_backend()
    print("backend file:", be.__file__)
    print("backend cuda attrs:", [x for x in dir(be) if "cuda" in x.lower()])
    if hasattr(be, "cuda_available"):
        print("cuda_available():", be.cuda_available())
    elif hasattr(be, "cuda_is_available"):
        print("cuda_is_available():", be.cuda_is_available())
    else:
        print("no cuda api exported")
except Exception as e:
    print("backend:", e)

print("\n== G6K ==")
try:
    import g6k

    print("g6k file:", g6k.__file__)
    from g6k import Siever, SieverParams

    p = SieverParams()
    knobs = [k for k in dir(p) if "gpu" in k.lower()]
    print("gpu knobs:", knobs if knobs else "NONE (this is CPU g6k)")
except Exception as e:
    print("g6k:", e)

print("\n== nvidia-smi ==")
os.system(
    "nvidia-smi --query-gpu=name,memory.used,utilization.gpu "
    "--format=csv,noheader || true"
)
