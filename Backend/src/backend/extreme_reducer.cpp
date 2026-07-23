#include "extreme_reducer.hpp"

#include "bkz2_engine.hpp"
#include "enumeration_engine.hpp"
#include "matrix_utils.hpp"
#include "sieve_bridge.hpp"

#include <chrono>
#include <cmath>
#include <exception>
#include <string>

namespace lattice_backend {

ReduceResult reduce_extreme(int64_t matrix_id, Matrix& B,
                            int pos, int beta, bool use_sieve) {
    ReduceResult out;
    const auto started = std::chrono::steady_clock::now();

    try {
        if (pos < 0 || pos >= B.get_rows() ||
            beta < kMinimumActionBeta || beta > kMaximumActionBeta) {
            out.stop_reason = StopReason::invalid_input;
            out.error = "invalid local reduction block";
            out.time_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            return out;
        }

        const int actual_beta = std::min(beta, B.get_rows() - pos);
        out.actual_beta = actual_beta;
        if (actual_beta < kMinimumActionBeta ||
            actual_beta > kMaximumActionBeta) {
            out.stop_reason = StopReason::invalid_input;
            out.error = "clipped local action beta is outside [21, 95]";
            out.time_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            return out;
        }

        Matrix original_block = copy_block(B, pos, actual_beta);
        Matrix candidate = original_block;
        const double potential_before = log_potential(original_block);

        if (!use_sieve) {
            out.backend = "bkz2_preconditioner";
            out.exact = true;
            out.completed = run_bkz2_preconditioner(
                candidate, PreconditionerProfile::normal, &out.error);
            out.stop_reason = out.completed ? StopReason::completed
                                            : StopReason::precondition_failed;
        } else if (actual_beta < kSieveThreshold) {
            out.backend = "local_bkz2_pruned_enumeration";
            const EnumerationRunInfo enumeration = run_extreme_enumeration(
                candidate, enumeration_budget_from_environment());
            out.completed = enumeration.completed;
            out.changed = enumeration.changed;
            out.exact = enumeration.exact;
            out.early_stopped = enumeration.early_stopped;
            out.quality_target_reached = enumeration.quality_target_reached;
            out.enumeration_rounds = enumeration.rounds;
            out.potential_drop_per_dimension =
                enumeration.potential_drop_per_dimension;
            out.first_gso_log_drop = enumeration.first_gso_log_drop;
            out.stop_reason = enumeration.stop_reason;
            out.error = enumeration.error;
        } else {
            out.backend = "local_bgj_sieve";

            if (!run_bkz2_preconditioner(
                    candidate, PreconditionerProfile::normal, &out.error)) {
                out.stop_reason = StopReason::precondition_failed;
            } else {
                const SieveRunInfo sieve =
                    run_local_extreme_sieve(candidate, matrix_id, pos);
                out.completed = sieve.completed;
                out.changed = sieve.changed;
                out.exact = sieve.exact;
                out.stop_reason = sieve.stop_reason;
                out.error = sieve.error;
                out.sieve_dimension = sieve.final_csd;
                out.dimension_for_free = sieve.dimension_for_free;
                out.bgj_calls = sieve.bgj_calls;
                out.database_vectors = sieve.vectors;

                if (out.completed && out.exact) {
                    std::string post_error;
                    if (!run_bkz2(candidate, std::min(30, actual_beta), 1, 1.0,
                                  &post_error)) {
                        out.completed = false;
                        out.stop_reason = StopReason::precondition_failed;
                        out.error = std::move(post_error);
                    } else {
                        fplll::lll_reduction(candidate, 0.999);
                    }
                }
            }
        }

        const double potential_after = log_potential(candidate);
        out.non_worsening = out.completed && out.exact &&
                            std::isfinite(potential_after) &&
                            potential_after <= potential_before + 1e-10;
        out.changed = out.completed && out.exact &&
                      !matrices_equal(candidate, original_block);
        out.accepted = out.changed && out.non_worsening;

        if (out.accepted) {
            if (!write_block(B, pos, candidate, &out.error)) {
                out.accepted = false;
                out.completed = false;
                out.stop_reason = StopReason::invalid_input;
            } else if (out.stop_reason == StopReason::none ||
                       out.stop_reason == StopReason::no_change) {
                out.stop_reason = StopReason::completed;
            }
        } else if (out.completed && !out.changed) {
            out.stop_reason = StopReason::no_change;
        } else if (out.completed && out.changed && !out.non_worsening) {
            out.stop_reason = StopReason::non_worsening_rejected;
            if (out.error.empty()) {
                out.error = "candidate block was exact but failed the non-worsening transaction gate";
            }
        }
    } catch (const std::exception& ex) {
        out.completed = false;
        out.accepted = false;
        out.stop_reason = StopReason::exception;
        out.error = ex.what();
    } catch (...) {
        out.completed = false;
        out.accepted = false;
        out.stop_reason = StopReason::exception;
        out.error = "local reduction failed with an unknown exception";
    }

    out.time_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return out;
}

}  // namespace lattice_backend
