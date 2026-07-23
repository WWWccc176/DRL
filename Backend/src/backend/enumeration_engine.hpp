#pragma once

#include "types.hpp"
#include <string>

namespace lattice_backend {

struct EnumerationRunInfo {
    bool completed = false;
    bool changed = false;
    bool exact = true;
    bool early_stopped = false;
    bool quality_target_reached = false;
    int rounds = 0;
    double potential_drop_per_dimension = 0.0;
    double first_gso_log_drop = 0.0;
    StopReason stop_reason = StopReason::none;
    std::string error;
};

EnumerationBudget enumeration_budget_from_environment();

EnumerationRunInfo run_extreme_enumeration(
    Matrix& B, const EnumerationBudget& budget = EnumerationBudget{});

// Compatibility wrapper retained for older callers.
inline bool run_extreme_enumeration(Matrix& B, int max_rounds) {
    EnumerationBudget budget;
    budget.max_rounds = max_rounds;
    return run_extreme_enumeration(B, budget).changed;
}

}  // namespace lattice_backend
