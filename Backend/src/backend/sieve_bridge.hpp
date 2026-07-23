#pragma once

#include "types.hpp"
#include <cstdint>
#include <string>

namespace lattice_backend {

struct SieveRunInfo {
    bool completed = false;
    bool changed = false;
    bool exact = false;
    bool inserted = false;
    StopReason stop_reason = StopReason::none;
    std::string error;

    int final_csd = 0;
    int dimension_for_free = 0;
    int bgj_calls = 0;
    long long vectors = 0;
};

SieveBudget sieve_budget_from_environment();

SieveRunInfo run_local_extreme_sieve(
    Matrix& block,
    int64_t matrix_id,
    int global_pos,
    const SieveBudget& budget = sieve_budget_from_environment());

// Call once when a persistent GPU worker is shutting down. It releases the
// engine's global pinned chunk allocator and resets the CUDA context.
void shutdown_local_sieve_engine();

}  // namespace lattice_backend
