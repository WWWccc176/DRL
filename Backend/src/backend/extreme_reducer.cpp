#include "extreme_reducer.hpp"
#include "bkz2_engine.hpp"
#include "enumeration_engine.hpp"
#include "matrix_utils.hpp"
#include "sieve_bridge.hpp"
#include <chrono>
#include <cmath>

static constexpr int SIEVE_THRESHOLD = 40;

namespace lattice_backend {
ReduceResult reduce_extreme(int64_t matrix_id, Matrix& B, int pos, int beta, bool bool_sieve) {
    ReduceResult out;
    const auto t0=std::chrono::steady_clock::now();
    if (pos<0 || pos>=B.get_rows() || beta<2) return out;
    const int ab=std::min(beta,B.get_rows()-pos);
    out.actual_beta=ab;
    Matrix block=copy_block(B,pos,ab);
    const double pot_before=log_potential(block);

    run_extreme_bkz2_preconditioner(block);

    if (!bool_sieve) {
        out.backend="bkz2_max";
    } else if (ab<SIEVE_THRESHOLD) {
        const bool changed=run_extreme_enumeration(block,std::min(48,std::max(12,ab)));
        out.backend="enumeration_exact";
        out.accepted=changed;
    } else {
        SieveRunInfo s=run_local_extreme_sieve(block,matrix_id,pos);
        out.backend="local_bgj_sieve";
        out.sieve_dimension=s.final_csd;
        out.database_vectors=s.vectors;
        out.accepted=s.changed;
        run_bkz2(block,std::min(45,ab),12,1.0);
        fplll::lll_reduction(block,0.999);
    }

    const double pot_after=log_potential(block);
    const bool improved=std::isfinite(pot_after) && pot_after<=pot_before+1e-10;
    if (improved) {
        write_block(B,pos,block);
        out.accepted=true;
    }
    out.time_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    return out;
}
}
