#pragma once

#include <fplll.h>
#include <cstdint>
#include <string>

namespace lattice_backend {

using Matrix = fplll::ZZ_mat<mpz_t>;

inline constexpr int kInitialBkzBeta = 20;
inline constexpr int kMinimumActionBeta = 21;
inline constexpr int kEnumerationMaxBeta = 52;
inline constexpr int kSieveThreshold = 53;
inline constexpr int kMaximumActionBeta = 95;
inline constexpr int kBgj3MinBeta = 95;

enum class StopReason {
    none,
    completed,
    quality_target_reached,
    stagnation,
    no_change,
    invalid_input,
    precondition_failed,
    enumeration_failed,
    sieve_failed,
    exact_recovery_failed,
    non_worsening_rejected,
    budget_exhausted,
    exception
};

inline const char* to_string(StopReason reason) {
    switch (reason) {
        case StopReason::none: return "none";
        case StopReason::completed: return "completed";
        case StopReason::quality_target_reached: return "quality_target_reached";
        case StopReason::stagnation: return "stagnation";
        case StopReason::no_change: return "no_change";
        case StopReason::invalid_input: return "invalid_input";
        case StopReason::precondition_failed: return "precondition_failed";
        case StopReason::enumeration_failed: return "enumeration_failed";
        case StopReason::sieve_failed: return "sieve_failed";
        case StopReason::exact_recovery_failed: return "exact_recovery_failed";
        case StopReason::non_worsening_rejected: return "non_worsening_rejected";
        case StopReason::budget_exhausted: return "budget_exhausted";
        case StopReason::exception: return "exception";
    }
    return "unknown";
}

enum class PreconditionerProfile {
    light,
    normal,
    strong
};

struct EnumerationBudget {
    int max_rounds = 2;
    double lll_delta = 0.999;
    double gh_factor = 1.05;
    double target_potential_drop_per_dimension = 5.0e-4;
    double target_first_gso_log_drop = 1.0e-3;
    double min_round_potential_drop_per_dimension = 1.0e-5;
};

struct SieveBudget {
    std::int64_t max_vectors = 2'000'000;
    int max_csd = 175;
    int max_bgj_calls = 1;
    double max_wall_seconds = 0.0;
    bool progressive = false;
    bool enable_dual_hash = true;
    bool cleanup_workdir = true;
    double insertion_eta = 1.05;
};

struct ReduceResult {
    std::string backend = "none";
    bool completed = false;
    bool changed = false;
    bool exact = false;
    bool non_worsening = false;
    bool accepted = false;
    bool early_stopped = false;
    bool quality_target_reached = false;
    StopReason stop_reason = StopReason::none;
    std::string error;

    int actual_beta = 0;
    int enumeration_rounds = 0;
    double potential_drop_per_dimension = 0.0;
    double first_gso_log_drop = 0.0;
    int sieve_dimension = 0;
    int dimension_for_free = 0;
    int bgj_calls = 0;
    long long database_vectors = 0;
    double time_ms = 0.0;
};

}  // namespace lattice_backend
