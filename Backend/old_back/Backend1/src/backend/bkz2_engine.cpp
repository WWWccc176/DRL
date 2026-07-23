#include "bkz2_engine.hpp"
#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace lattice_backend {
static constexpr int ENUMERATION_BKZ_CEILING=45;
static bool g_loaded=false;
static std::vector<fplll::Strategy> g_strategies;
static std::once_flag g_once;

static std::vector<fplll::Strategy>& strategies() {
    std::call_once(g_once,[](){
        const char* p=std::getenv("FPLLL_STRATEGIES_JSON");
        if (p && *p) try { g_strategies=fplll::load_strategies_json(p); } catch(...) {}
        if (g_strategies.empty()) try {
            std::string p2=fplll::strategy_full_path("default.json");
            if (!p2.empty()) g_strategies=fplll::load_strategies_json(p2);
        } catch(...) {}
        for (long b=(long)g_strategies.size(); b<=ENUMERATION_BKZ_CEILING; ++b)
            g_strategies.emplace_back(fplll::Strategy::EmptyStrategy(b));
        g_loaded=true;
    });
    return g_strategies;
}

void run_bkz2(Matrix& B, int block_size, int loops, double gh_factor) {
    if (B.get_rows()<2) return;
    block_size=std::max(2,std::min({block_size,B.get_rows(),ENUMERATION_BKZ_CEILING}));
    fplll::BKZParam par(block_size,strategies());
    par.flags=fplll::BKZ_AUTO_ABORT|fplll::BKZ_GH_BND|fplll::BKZ_MAX_LOOPS;
    par.max_loops=std::max(1,loops);
    par.gh_factor=gh_factor;
    try { fplll::bkz_reduction(&B,nullptr,par); } catch(...) {}
}

void run_extreme_bkz2_preconditioner(Matrix& B) {
    if (B.get_rows()<2) return;
    fplll::lll_reduction(B,0.999);
    const int d=B.get_rows();
    const int stages[]={20,30,40,45};
    const int loops[]={8,12,16,24};
    for (int i=0;i<4;++i) if (d>=2) run_bkz2(B,std::min(d,stages[i]),loops[i],1.0);
    fplll::lll_reduction(B,0.999);
}
}
