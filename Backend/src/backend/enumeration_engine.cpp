#include "enumeration_engine.hpp"
#include "bkz2_engine.hpp"
#include "matrix_utils.hpp"
#include <algorithm>
#include <vector>

namespace lattice_backend {
bool run_extreme_enumeration(Matrix& B, int max_rounds) {
    if (B.get_rows()<2) return false;
    run_extreme_bkz2_preconditioner(B);
    bool any=false;
    double best_pot=log_potential(B);
    for (int round=0; round<std::max(1,max_rounds); ++round) {
        std::vector<fplll::Z_NR<mpz_t>> sol;
        int st=-1;
        try { st=fplll::shortest_vector(B,sol); } catch(...) { break; }
        if (st!=fplll::RED_SUCCESS || (int)sol.size()!=B.get_rows()) break;
        bool nonzero=false; for (auto& z:sol) if (z.sgn()!=0) {nonzero=true;break;}
        if (!nonzero) break;
        Matrix snapshot=B;
        insert_and_lll(B,sol,0.999);
        run_bkz2(B,std::min(45,B.get_rows()),8,1.0);
        const double p=log_potential(B);
        if (p<best_pot-1e-10) { best_pot=p; any=true; }
        else { B=std::move(snapshot); break; }
    }
    return any;
}
}
