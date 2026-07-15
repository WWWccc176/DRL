# /tmp/poison_test.py
import os, sys

os.environ["LATTICE_DISABLE_CUDA"] = "1"
sys.path.insert(0, "/home/amax/projects/DRL")
import faulthandler

faulthandler.enable()
import my_project_backend as B


def load(fp):
    rows = []
    for line in open(fp).read().replace("[", " ").replace("]", " ").splitlines():
        if line.strip():
            rows.append(line.split())
    return "[" + "\n".join("[" + " ".join(r) + "]" for r in rows) + "]"


mid = B.create_matrix_lll(
    load("/home/amax/projects/DRL/dataset/svpchallengedim60seed0.txt")
)
if len(sys.argv) > 1 and sys.argv[1] == "poison":
    B.reduce(mid, "LOCAL_BKZ", 8, 0)  # 先用小 blocksize 播种 static vector
    print("poisoned with bs=8", flush=True)
for k in range(30):  # 多打几次，提高越界读踩到垃圾的概率
    B.reduce(mid, "ORACLE_ENUM_BLOCK", 40, (k * 3) % 20)
    print("enum-40 round", k, "ok", flush=True)
print("ALL OK", flush=True)
