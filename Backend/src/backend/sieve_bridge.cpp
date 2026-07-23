#include "sieve_bridge.hpp"

#include "../sieve_engine/include/pool_hd.h"
#include "matrix_utils.hpp"

#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

namespace fs = std::filesystem;

namespace lattice_backend {
namespace {

std::mutex g_sieve_engine_mutex;
std::atomic<std::uint64_t> g_invocation_id{1};
bool g_engine_shutdown = false;

NTL::ZZ mpz_to_zz(const mpz_t value) {
    char *raw = mpz_get_str(nullptr, 10, value);
    NTL::ZZ out;
    std::stringstream stream(raw ? raw : "0");
    stream >> out;
    if (raw) {
        void (*free_function)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &free_function);
        free_function(raw, std::strlen(raw) + 1);
    }
    return out;
}

long long environment_integer(const char *name, long long fallback,
                              long long minimum, long long maximum) {
    const char *raw = std::getenv(name);
    if (!raw || !*raw)
        return fallback;
    char *end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (end == raw || *end != '\0')
        return fallback;
    return std::max(minimum, std::min(maximum, parsed));
}

double environment_double(const char *name, double fallback, double minimum,
                          double maximum) {
    const char *raw = std::getenv(name);
    if (!raw || !*raw)
        return fallback;
    char *end = nullptr;
    const double parsed = std::strtod(raw, &end);
    if (end == raw || *end != '\0' || !std::isfinite(parsed))
        return fallback;
    return std::max(minimum, std::min(maximum, parsed));
}

bool environment_bool(const char *name, bool fallback) {
    const char *raw = std::getenv(name);
    if (!raw || !*raw)
        return fallback;
    if (*raw == '1' || *raw == 't' || *raw == 'T' || *raw == 'y' ||
        *raw == 'Y') {
        return true;
    }
    if (*raw == '0' || *raw == 'f' || *raw == 'F' || *raw == 'n' ||
        *raw == 'N') {
        return false;
    }
    return fallback;
}

class WorkDirectoryCleanup {
  public:
    WorkDirectoryCleanup(fs::path path, bool enabled)
        : path_(std::move(path)), enabled_(enabled) {}

    WorkDirectoryCleanup(const WorkDirectoryCleanup &) = delete;
    WorkDirectoryCleanup &operator=(const WorkDirectoryCleanup &) = delete;

    ~WorkDirectoryCleanup() {
        if (!enabled_)
            return;
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

  private:
    fs::path path_;
    bool enabled_;
};

class CurrentDirectoryGuard {
  public:
    explicit CurrentDirectoryGuard(const fs::path &target)
        : old_(fs::current_path()) {
        fs::create_directories(target);
        fs::current_path(target);
    }

    CurrentDirectoryGuard(const CurrentDirectoryGuard &) = delete;
    CurrentDirectoryGuard &operator=(const CurrentDirectoryGuard &) = delete;

    ~CurrentDirectoryGuard() {
        std::error_code ignored;
        fs::current_path(old_, ignored);
    }

  private:
    fs::path old_;
};

int run_best_bgj(Pool_hd_t &pool, int action_beta) {
    // The learned block size is the routing contract. A 95-row local block is
    // represented by CSD=94 inside Pool_hd_t, so routing on CSD would start
    // BGJ3 one action too late.
    if (action_beta < kBgj3MinBeta)
        return pool.bgj2_Sieve_hd();
    return pool.bgj3_Sieve_hd();
}

long target_database_size(int csd, const SieveBudget &budget) {
    long double expected =
        3.2L * std::pow(4.0L / 3.0L, static_cast<long double>(csd) * 0.5L) -
        5.0L;
    expected = std::min(expected, static_cast<long double>(budget.max_vectors));
    expected = std::max(expected, static_cast<long double>(csd + 64));
    expected = std::min(
        expected, static_cast<long double>(std::numeric_limits<long>::max()));
    return static_cast<long>(expected);
}

fs::path make_work_directory(const Matrix &block, int64_t matrix_id,
                             int global_pos) {
    const char *configured = std::getenv("LATTICE_SIEVE_WORKDIR");
    const fs::path base = configured && *configured
                              ? fs::path(configured)
                              : fs::path("/tmp/my_project_sieve");

    std::ostringstream name;
    name << matrix_id << '_' << global_pos << '_' << block.get_rows() << '_'
         << std::hex << matrix_fingerprint(block) << '_'
         << g_invocation_id.fetch_add(1, std::memory_order_relaxed);
    return base / name.str();
}

} // namespace

SieveBudget sieve_budget_from_environment() {
    SieveBudget budget;
    budget.max_vectors = environment_integer(
        "LATTICE_SIEVE_MAX_VECTORS", budget.max_vectors, 128, 500'000'000);
    budget.max_csd = static_cast<int>(
        environment_integer("LATTICE_SIEVE_MAX_CSD", budget.max_csd, 39, 175));
    budget.max_bgj_calls = 1;
    budget.max_wall_seconds = environment_double(
        "LATTICE_SIEVE_MAX_SECONDS", budget.max_wall_seconds, 0.0, 86400.0);
    budget.progressive = false;
    budget.enable_dual_hash =
        environment_bool("LATTICE_SIEVE_ENABLE_DH", budget.enable_dual_hash);
    budget.cleanup_workdir =
        environment_bool("LATTICE_SIEVE_CLEANUP", budget.cleanup_workdir);
    budget.insertion_eta = environment_double("LATTICE_SIEVE_INSERT_ETA",
                                              budget.insertion_eta, 1.0, 2.0);
    return budget;
}

SieveRunInfo run_local_extreme_sieve(Matrix &block, int64_t matrix_id,
                                     int global_pos,
                                     const SieveBudget &budget) {
    SieveRunInfo info;
    const int dimension = block.get_rows();
    const int cols = block.get_cols();
    if (dimension < 40 || dimension > kMaximumActionBeta || cols < dimension) {
        info.stop_reason = StopReason::invalid_input;
        info.error = "BGJ requires a full-row-rank block with 40-95 rows";
        return info;
    }

    std::unique_lock<std::mutex> engine_lock(g_sieve_engine_mutex);
    if (g_engine_shutdown) {
        info.stop_reason = StopReason::sieve_failed;
        info.error = "the local sieve engine has already been shut down";
        return info;
    }

    const Matrix original = block;
    const fs::path work_directory =
        make_work_directory(original, matrix_id, global_pos);
    const auto started = std::chrono::steady_clock::now();
    WorkDirectoryCleanup cleanup(work_directory, budget.cleanup_workdir);

    try {
        {
            CurrentDirectoryGuard cwd(work_directory);

            NTL::Mat<NTL::ZZ> exact_input;
            exact_input.SetDims(dimension, cols);
            for (int i = 0; i < dimension; ++i) {
                for (int j = 0; j < cols; ++j) {
                    exact_input[i][j] = mpz_to_zz(original[i][j].get_data());
                }
            }

            Lattice_QP lattice(exact_input);
            lattice.LLL_QP(0.999);

            const int maximum_csd =
                std::max(39, std::min({dimension - 1, budget.max_csd, 175}));
            int starting_csd = maximum_csd;
            if (budget.progressive) {
                starting_csd = std::min(
                    maximum_csd, std::max(39, std::min(60, maximum_csd)));
            }

            Pool_hd_t pool(&lattice);
            pool.set_sieving_context(dimension - starting_csd, dimension);
            pool.set_boost_depth(0);

            const unsigned hardware_threads =
                std::thread::hardware_concurrency();
            const unsigned default_threads =
                hardware_threads ? hardware_threads : 8U;
            const int requested_threads = static_cast<int>(environment_integer(
                "LATTICE_SIEVE_THREADS",
                std::max(4U, std::min(32U, default_threads)), 1, 256));
            pool.set_num_threads(requested_threads);
            pool.sampling(target_database_size(pool.CSD, budget));

            int last_return = 0;
            while (info.bgj_calls < budget.max_bgj_calls) {
                last_return = run_best_bgj(pool, dimension);
                ++info.bgj_calls;

                if (budget.max_wall_seconds > 0.0) {
                    const double elapsed =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - started)
                            .count();
                    if (elapsed >= budget.max_wall_seconds) {
                        info.stop_reason = StopReason::budget_exhausted;
                        break;
                    }
                }

                if (!budget.progressive || pool.CSD >= maximum_csd)
                    break;
                if (last_return == -1 && pool.CSD <= 100 &&
                    pool.check_dim_lose() == -1) {
                    break;
                }
                pool.extend_left();
            }

            pool.down_sieve_flag = 1;
            long inserted_position = -1;
            int insertion_status = -1;
            if (budget.enable_dual_hash && pool.CSD >= 60) {
                insertion_status = pool.dh_insert(0, budget.insertion_eta, 0.0,
                                                  &inserted_position, 0.0);
            }
            if (insertion_status != 0) {
                insertion_status =
                    pool.insert(0, budget.insertion_eta, &inserted_position, 1);
            }
            info.inserted = insertion_status == 0 && inserted_position >= 0;

            if (dimension < 90)
                lattice.LLL_DEEP_QP(0.999);
            lattice.LLL_QP(0.999);

            // Do not call Lattice_QP::to_int(): it rounds only the high double
            // component and can discard significant low bits. Recover from the
            // complete hi+lo value and verify the exact unimodular transform.
            MAT_QP approximate = lattice.get_b();
            Matrix exact_recovered;
            if (!recover_exact_unimodular_basis(original, approximate.hi,
                                                approximate.lo, dimension, cols,
                                                exact_recovered, &info.error)) {
                info.stop_reason = StopReason::exact_recovery_failed;
                return info;
            }

            const int final_csd = static_cast<int>(pool.CSD);
            info.final_csd = final_csd;
            info.dimension_for_free = std::max(0, dimension - final_csd);
            info.vectors = pool.pwc_manager ? pool.pwc_manager->num_vec() : 0;
            info.exact = true;
            info.completed = true;
            info.changed = !matrices_equal(original, exact_recovered);
            info.stop_reason =
                info.changed ? StopReason::completed : StopReason::no_change;
            block = std::move(exact_recovered);
        }
    } catch (const std::exception &ex) {
        info.stop_reason = StopReason::exception;
        info.error = std::string("BGJ bridge exception: ") + ex.what();
    } catch (...) {
        info.stop_reason = StopReason::exception;
        info.error = "BGJ bridge failed with an unknown exception";
    }

    return info;
}

void shutdown_local_sieve_engine() {
    std::lock_guard<std::mutex> lock(g_sieve_engine_mutex);
    if (g_engine_shutdown)
        return;
    _destory_ck_allocator();
    g_engine_shutdown = true;
}

} // namespace lattice_backend
