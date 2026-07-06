#!/usr/bin/env python3
"""
ENUM+SIEVE oracle:
    beta < 50           -> LOCAL_BKZ (small blocks; enum/sieve not worth it)
    50 <= beta < 60     -> ORACLE_ENUM_BLOCK (pure enumeration)
    beta >= 60          -> G6K-GPU sieve; on ANY failure fall back to
                           ORACLE_ENUM_BLOCK (never BKZ)

GPU path: dump_block(original rows) -> Siever.gpu_sieve() -> best_lifts()[0]
gives integer coeffs w.r.t. the ORIGINAL block rows -> insert_coeff_vector.
No LLL pre-reduction, no Babai, no self-written sieve.

MUST run in a dedicated single worker (g6k_task.py / g6k_server.py).
"""

# chdir into g6k root (holds ./spherical_coding/) before importing g6k anywhere.
import g6k_env  # noqa: F401  (import for side effect)

from lattice_metrics import parse_fplll

_G6K_OK = None


def g6k_available():
    global _G6K_OK
    if _G6K_OK is None:
        try:
            import g6k  # noqa
            import fpylll  # noqa

            _G6K_OK = True
        except Exception:
            _G6K_OK = False
    return _G6K_OK


def g6k_reduce_block(be, mid, pos, beta, threads=1, min_enum=50, min_sieve=60):
    # beta < 50: small block -> cheap CPU BKZ fallback
    if beta < min_enum:
        return bool(be.reduce(mid, "LOCAL_BKZ", beta, pos).get("accepted"))

    # 50 <= beta < 60, or GPU/g6k unavailable: pure enumeration
    if beta < min_sieve or not g6k_available():
        return bool(be.reduce(mid, "ORACLE_ENUM_BLOCK", beta, pos).get("accepted"))

    from fpylll import IntegerMatrix
    from g6k import Siever, SieverParams

    # ORIGINAL block rows (NO LLL here: coeffs must be w.r.t. these rows)
    B0 = parse_fplll(be.dump_block(mid, pos, beta))
    A = IntegerMatrix.from_matrix([[int(x) for x in r] for r in B0])

    try:
        params = SieverParams(
            threads=threads,
            gpus=1,
            gpu_bucketer=b"bdgl",
            gpu_triple=False,
        )
    except Exception:
        try:
            params = SieverParams(threads=threads)
        except Exception:
            params = None

    try:
        g6k = Siever(A, params) if params is not None else Siever(A)
        g6k.initialize_local(0, 0, A.nrows)
        if hasattr(g6k, "gpu_sieve"):
            g6k.gpu_sieve()
        else:
            g6k(alg="bgj1")  # CPU fallback if GPU sieve absent
    except Exception:
        # sieve failed -> enumeration, NOT BKZ
        return bool(be.reduce(mid, "ORACLE_ENUM_BLOCK", beta, pos).get("accepted"))

    # best_lifts(): (index, norm, coeffs) w.r.t. the Siever basis (= B0)
    try:
        lifts = g6k.best_lifts()
    except Exception:
        lifts = None
    if not lifts:
        return bool(be.reduce(mid, "ORACLE_ENUM_BLOCK", beta, pos).get("accepted"))

    best = min(lifts, key=lambda t: t[1])  # shortest lift
    coeffs = best[2]
    if coeffs is None or len(coeffs) == 0:
        return False

    return bool(
        be.insert_coeff_vector(mid, pos, beta, [str(int(x)) for x in coeffs]).get(
            "accepted"
        )
    )

