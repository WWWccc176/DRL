#include "bkz2_engine.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace lattice_backend {
namespace {

constexpr int kEnumerationBkzCeiling = kMaximumActionBeta;
std::vector<fplll::Strategy> g_strategies;
std::once_flag g_strategy_once;

std::vector<fplll::Strategy>& strategies() {
    std::call_once(g_strategy_once, []() {
        const char* configured = std::getenv("FPLLL_STRATEGIES_JSON");
        if (configured && *configured) {
            try {
                g_strategies = fplll::load_strategies_json(configured);
            } catch (...) {
                g_strategies.clear();
            }
        }

        if (g_strategies.empty()) {
            try {
                const std::string path = fplll::strategy_full_path("default.json");
                if (!path.empty()) g_strategies = fplll::load_strategies_json(path);
            } catch (...) {
                g_strategies.clear();
            }
        }

        for (long block = static_cast<long>(g_strategies.size());
             block <= kEnumerationBkzCeiling; ++block) {
            g_strategies.emplace_back(fplll::Strategy::EmptyStrategy(block));
        }
    });
    return g_strategies;
}

bool set_error(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

}  // namespace

bool run_bkz2(Matrix& B, int block_size, int loops, double gh_factor,
              std::string* error) {
    if (B.get_rows() < 2) return true;

    block_size = std::max(
        2, std::min({block_size, B.get_rows(), kEnumerationBkzCeiling}));
    loops = std::max(1, loops);

    try {
        fplll::BKZParam parameters(block_size, strategies());
        parameters.flags = fplll::BKZ_AUTO_ABORT |
                           fplll::BKZ_GH_BND |
                           fplll::BKZ_MAX_LOOPS;
        parameters.max_loops = loops;
        parameters.gh_factor = gh_factor;
        fplll::bkz_reduction(&B, nullptr, parameters);
        return true;
    } catch (const std::exception& ex) {
        return set_error(error, std::string("BKZ 2.0 failed: ") + ex.what());
    } catch (...) {
        return set_error(error, "BKZ 2.0 failed with an unknown exception");
    }
}


bool run_local_pruned_svp_round(Matrix& B, double lll_delta,
                                double gh_factor, std::string* error) {
    const int block_size = B.get_rows();
    if (block_size < 2 || block_size > kEnumerationBkzCeiling) {
        return set_error(error, "local pruned SVP block dimension is out of range");
    }

    try {
        // Use fplll's strategy-selected local preprocessing and pruning. Any
        // exact row transformations created while preprocessing this projected
        // block remain in B, rather than retaining only the final short vector.
        // This is the local-basis-processing improvement described by Zhao-Ding.
        std::vector<fplll::Strategy> local_strategies = strategies();

        fplll::BKZParam parameters(block_size, local_strategies, lll_delta);
        parameters.flags = fplll::BKZ_NO_LLL |
                           fplll::BKZ_GH_BND |
                           fplll::BKZ_BOUNDED_LLL;
        parameters.gh_factor = gh_factor;

        using Integer = fplll::Z_NR<mpz_t>;
        using Float = fplll::FP_NR<double>;
        Matrix transform;
        Matrix inverse_transform;
        fplll::MatGSO<Integer, Float> gso(
            B, transform, inverse_transform, fplll::GSO_ROW_EXPO);
        fplll::LLLReduction<Integer, Float> lll(
            gso, lll_delta, fplll::LLL_DEF_ETA, fplll::LLL_DEFAULT);
        fplll::BKZReduction<Integer, Float> reducer(gso, lll, parameters);

        gso.discover_all_rows();
        (void)reducer.svp_reduction(0, block_size, parameters);

        // svp_reduction preserves exact integer row operations but does not
        // promise that the complete output block is LLL-reduced.
        if (!lll.lll(0, 0, B.get_rows(), 0)) {
            return set_error(error, "post-SVP local LLL failed");
        }
        return true;
    } catch (const std::exception& ex) {
        return set_error(error,
                         std::string("local pruned SVP round failed: ") + ex.what());
    } catch (...) {
        return set_error(error,
                         "local pruned SVP round failed with an unknown exception");
    }
}

bool run_bkz2_preconditioner(Matrix& B, PreconditionerProfile profile,
                             std::string* error) {
    if (B.get_rows() < 2) return true;

    try {
        fplll::lll_reduction(B, 0.999);
    } catch (const std::exception& ex) {
        return set_error(error, std::string("LLL preconditioner failed: ") + ex.what());
    } catch (...) {
        return set_error(error, "LLL preconditioner failed with an unknown exception");
    }

    const int dimension = B.get_rows();
    std::vector<std::pair<int, int>> stages;

    switch (profile) {
        case PreconditionerProfile::light:
            stages = {{20, 1}};
            break;
        case PreconditionerProfile::normal:
            stages = {{20, 2}, {30, 2}};
            break;
        case PreconditionerProfile::strong:
            stages = {{20, 2}, {30, 3}, {40, 3}, {45, 4}};
            break;
    }

    for (const auto& [block_size, loops] : stages) {
        if (dimension < 2) break;
        if (!run_bkz2(B, std::min(dimension, block_size), loops, 1.0, error)) {
            return false;
        }
    }

    try {
        fplll::lll_reduction(B, 0.999);
        return true;
    } catch (const std::exception& ex) {
        return set_error(error, std::string("Final LLL preconditioner failed: ") + ex.what());
    } catch (...) {
        return set_error(error, "Final LLL preconditioner failed with an unknown exception");
    }
}

}  // namespace lattice_backend
