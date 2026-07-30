#include "enumeration_engine.hpp"

#include "bkz2_engine.hpp"
#include "matrix_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <string>
#include <utility>

namespace lattice_backend {
namespace {

long long environment_integer(const char* name, long long fallback,
                              long long minimum, long long maximum) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (end == raw || *end != '\0') return fallback;
    return std::max(minimum, std::min(maximum, parsed));
}

double environment_double(const char* name, double fallback,
                          double minimum, double maximum) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    const double parsed = std::strtod(raw, &end);
    if (end == raw || *end != '\0' || !std::isfinite(parsed)) return fallback;
    return std::max(minimum, std::min(maximum, parsed));
}

bool run_local_lll(Matrix& B, double delta, std::string* error) {
    try {
        fplll::lll_reduction(B, delta);
        return true;
    } catch (const std::exception& ex) {
        if (error) *error = std::string("local LLL failed: ") + ex.what();
        return false;
    } catch (...) {
        if (error) *error = "local LLL failed with an unknown exception";
        return false;
    }
}

void update_quality_metrics(const Matrix& B, double initial_potential,
                            double initial_first_gso,
                            EnumerationRunInfo& out) {
    const int dimension = std::max(1, B.get_rows());
    const double current_potential = log_potential(B);
    const double current_first_gso = first_gso_log_norm(B);
    out.potential_drop_per_dimension =
        std::max(0.0, (initial_potential - current_potential) /
                          static_cast<double>(dimension));
    out.first_gso_log_drop =
        std::max(0.0, initial_first_gso - current_first_gso);
}

bool quality_target_reached(const EnumerationRunInfo& out,
                            const EnumerationBudget& budget) {
    return (budget.target_potential_drop_per_dimension > 0.0 &&
            out.potential_drop_per_dimension >=
                budget.target_potential_drop_per_dimension) ||
           (budget.target_first_gso_log_drop > 0.0 &&
            out.first_gso_log_drop >= budget.target_first_gso_log_drop);
}

}  // namespace

EnumerationBudget enumeration_budget_from_environment() {
    EnumerationBudget budget;
    budget.max_rounds = static_cast<int>(environment_integer(
        "LATTICE_ENUM_MAX_ROUNDS", budget.max_rounds, 1, 4));
    budget.lll_delta = environment_double(
        "LATTICE_ENUM_LLL_DELTA", budget.lll_delta, 0.75, 0.999999);
    budget.gh_factor = environment_double(
        "LATTICE_ENUM_GH_FACTOR", budget.gh_factor, 1.0, 2.0);
    budget.target_potential_drop_per_dimension = environment_double(
        "LATTICE_ENUM_TARGET_POT_DROP_PER_DIM",
        budget.target_potential_drop_per_dimension, 0.0, 1.0);
    budget.target_first_gso_log_drop = environment_double(
        "LATTICE_ENUM_TARGET_FIRST_GSO_LOG_DROP",
        budget.target_first_gso_log_drop, 0.0, 1.0);
    budget.min_round_potential_drop_per_dimension = environment_double(
        "LATTICE_ENUM_MIN_ROUND_POT_DROP_PER_DIM",
        budget.min_round_potential_drop_per_dimension, 0.0, 1.0);
    return budget;
}

EnumerationRunInfo run_extreme_enumeration(
    Matrix& B, const EnumerationBudget& budget) {
    EnumerationRunInfo out;
    const int dimension = B.get_rows();
    if (dimension < kMinimumActionBeta || dimension > kEnumerationMaxBeta) {
        out.stop_reason = StopReason::invalid_input;
        out.error = "local enumeration action requires beta in [21, 52]";
        return out;
    }

    Matrix original = B;
    const double initial_potential = log_potential(original);
    const double initial_first_gso = first_gso_log_norm(original);
    if (!std::isfinite(initial_potential) || !std::isfinite(initial_first_gso)) {
        out.stop_reason = StopReason::invalid_input;
        out.error = "local block has non-finite initial GSO metrics";
        return out;
    }

    if (!run_local_lll(B, budget.lll_delta, &out.error)) {
        B = std::move(original);
        out.stop_reason = StopReason::precondition_failed;
        return out;
    }

    double previous_potential = log_potential(B);
    if (!std::isfinite(previous_potential)) {
        B = std::move(original);
        out.stop_reason = StopReason::precondition_failed;
        out.error = "local LLL produced a non-finite GSO profile";
        return out;
    }

    bool enumeration_improved = false;
    StopReason natural_stop = StopReason::completed;

    for (int round = 0; round < std::max(1, budget.max_rounds); ++round) {
        Matrix before_round = B;
        std::string round_error;

        if (!run_local_pruned_svp_round(
                B, budget.lll_delta, budget.gh_factor, &round_error)) {
            B = std::move(original);
            out.stop_reason = StopReason::enumeration_failed;
            out.error = std::move(round_error);
            return out;
        }

        const double current_potential = log_potential(B);
        const double current_first_gso = first_gso_log_norm(B);
        if (!std::isfinite(current_potential) ||
            !std::isfinite(current_first_gso)) {
            B = std::move(original);
            out.stop_reason = StopReason::enumeration_failed;
            out.error = "local pruned enumeration produced non-finite GSO metrics";
            return out;
        }

        const bool round_changed = !matrices_equal(B, before_round);
        const double round_drop_per_dimension =
            (previous_potential - current_potential) /
            static_cast<double>(dimension);

        if (current_potential > previous_potential + 1e-10) {
            B = std::move(before_round);
            natural_stop = StopReason::stagnation;
            out.early_stopped = true;
            break;
        }

        if (!round_changed ||
            round_drop_per_dimension <=
                budget.min_round_potential_drop_per_dimension) {
            B = std::move(before_round);
            natural_stop = StopReason::stagnation;
            out.early_stopped = round + 1 < budget.max_rounds;
            break;
        }

        enumeration_improved = true;
        ++out.rounds;
        previous_potential = current_potential;
        update_quality_metrics(B, initial_potential, initial_first_gso, out);

        if (quality_target_reached(out, budget)) {
            out.quality_target_reached = true;
            out.early_stopped = round + 1 < budget.max_rounds;
            natural_stop = StopReason::quality_target_reached;
            break;
        }
    }

    if (!enumeration_improved) {
        B = std::move(original);
        out.completed = true;
        out.changed = false;
        out.stop_reason = StopReason::no_change;
        out.potential_drop_per_dimension = 0.0;
        out.first_gso_log_drop = 0.0;
        return out;
    }

    const double final_potential = log_potential(B);
    const bool non_worsening = std::isfinite(final_potential) &&
                               final_potential <= initial_potential + 1e-10;
    const bool changed = !matrices_equal(B, original);

    out.completed = true;
    out.changed = changed && non_worsening;
    out.stop_reason = out.changed ? natural_stop : StopReason::no_change;

    if (!non_worsening) {
        B = std::move(original);
        out.changed = false;
        out.stop_reason = StopReason::non_worsening_rejected;
        out.error =
            "local pruned enumeration failed the exact MPZ potential transaction gate";
    }

    return out;
}

}  // namespace lattice_backend
