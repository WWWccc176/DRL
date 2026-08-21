#pragma once

#include "types.hpp"
#include <cstddef>
#include <string>

namespace lattice_backend {

bool run_bkz2(Matrix& B, int block_size, int loops, double gh_factor = 1.0,
              std::string* error = nullptr);

bool strategies_loaded_from_json();
std::string strategies_source_path();
std::size_t strategies_count();

bool run_bkz2_preconditioner(Matrix& B, PreconditionerProfile profile,
                             std::string* error = nullptr);

// Run exactly one pruned SVP reduction on the first local block and retain
// every exact basis transformation produced by preprocessing/postprocessing.
bool run_local_pruned_svp_round(Matrix& B, double lll_delta,
                                double gh_factor,
                                std::string* error = nullptr);

// Compatibility wrapper retained for older callers.
inline void run_extreme_bkz2_preconditioner(Matrix& B) {
    std::string ignored;
    (void)run_bkz2_preconditioner(B, PreconditionerProfile::strong, &ignored);
}

}  // namespace lattice_backend
