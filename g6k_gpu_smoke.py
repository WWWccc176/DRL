from fpylll import IntegerMatrix, LLL
from g6k import Siever, SieverParams

A = IntegerMatrix.random(80, "qary", k=40, bits=30)
LLL.reduction(A)

params = SieverParams(
    gpus=1,
    gpu_bucketer=b"bdgl",
)

g6k = Siever(A, params)
g6k.initialize_local(0, 0, 80)

print("before gpu_sieve")
g6k.gpu_sieve()
print("after gpu_sieve")

lifts = g6k.best_lifts()
print("num lifts:", len(lifts))
if lifts:
    print("best lift:", lifts[0])
