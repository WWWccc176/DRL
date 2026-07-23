#include "matrix_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <gmpxx.h>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lattice_backend {
namespace {

constexpr double kLog2 = 0.69314718055994530942;

bool fail(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

mpz_class to_mpz_class(const fplll::Z_NR<mpz_t>& value) {
    mpz_class out;
    mpz_set(out.get_mpz_t(), value.get_data());
    return out;
}

void assign_mpz(fplll::Z_NR<mpz_t>& destination, const mpz_class& value) {
    mpz_set(destination.get_data(), value.get_mpz_t());
}

bool round_double_double(double hi, double lo, mpz_class& rounded) {
    if (!std::isfinite(hi) || !std::isfinite(lo)) return false;

    mpq_class q_hi;
    mpq_class q_lo;
    mpq_set_d(q_hi.get_mpq_t(), hi);
    mpq_set_d(q_lo.get_mpq_t(), lo);
    mpq_class value = q_hi + q_lo;
    value.canonicalize();

    const mpz_class numerator = value.get_num();
    const mpz_class denominator = value.get_den();
    mpz_class floor_value;
    mpz_fdiv_q(floor_value.get_mpz_t(), numerator.get_mpz_t(), denominator.get_mpz_t());
    const mpz_class remainder = numerator - floor_value * denominator;

    rounded = floor_value;
    if (2 * remainder >= denominator) ++rounded;

    mpq_class difference = value - mpq_class(rounded);
    if (difference < 0) difference = -difference;

    // The original implementation considered errors above 0.2 fatal. 1/5 is
    // retained here, but evaluated against hi+lo exactly rather than hi alone.
    return difference <= mpq_class(1, 5);
}

std::uint64_t mod_pow(std::uint64_t base, std::uint64_t exponent,
                      std::uint64_t modulus) {
    std::uint64_t result = 1;
    while (exponent) {
        if (exponent & 1U) {
            result = static_cast<std::uint64_t>(
                (static_cast<unsigned __int128>(result) * base) % modulus);
        }
        base = static_cast<std::uint64_t>(
            (static_cast<unsigned __int128>(base) * base) % modulus);
        exponent >>= 1U;
    }
    return result;
}

bool find_independent_columns(const Matrix& matrix, std::vector<int>& pivots) {
    const int rows = matrix.get_rows();
    const int cols = matrix.get_cols();
    if (rows <= 0 || cols < rows) return false;

    constexpr std::array<std::uint64_t, 3> primes = {
        1'000'000'007ULL, 1'000'000'009ULL, 998'244'353ULL};

    for (const std::uint64_t prime : primes) {
        std::vector<std::uint64_t> work(static_cast<std::size_t>(rows) * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                work[static_cast<std::size_t>(i) * cols + j] =
                    mpz_fdiv_ui(matrix[i][j].get_data(), prime);
            }
        }

        pivots.clear();
        int pivot_row = 0;
        for (int column = 0; column < cols && pivot_row < rows; ++column) {
            int selected = pivot_row;
            while (selected < rows &&
                   work[static_cast<std::size_t>(selected) * cols + column] == 0) {
                ++selected;
            }
            if (selected == rows) continue;

            if (selected != pivot_row) {
                for (int j = 0; j < cols; ++j) {
                    std::swap(work[static_cast<std::size_t>(selected) * cols + j],
                              work[static_cast<std::size_t>(pivot_row) * cols + j]);
                }
            }

            const std::uint64_t pivot =
                work[static_cast<std::size_t>(pivot_row) * cols + column];
            const std::uint64_t inverse = mod_pow(pivot, prime - 2, prime);
            for (int j = column; j < cols; ++j) {
                work[static_cast<std::size_t>(pivot_row) * cols + j] =
                    static_cast<std::uint64_t>(
                        (static_cast<unsigned __int128>(
                             work[static_cast<std::size_t>(pivot_row) * cols + j]) *
                         inverse) % prime);
            }

            for (int i = 0; i < rows; ++i) {
                if (i == pivot_row) continue;
                const std::uint64_t factor =
                    work[static_cast<std::size_t>(i) * cols + column];
                if (factor == 0) continue;
                for (int j = column; j < cols; ++j) {
                    const std::uint64_t product = static_cast<std::uint64_t>(
                        (static_cast<unsigned __int128>(factor) *
                         work[static_cast<std::size_t>(pivot_row) * cols + j]) %
                        prime);
                    auto& entry = work[static_cast<std::size_t>(i) * cols + j];
                    entry = entry >= product ? entry - product
                                             : entry + prime - product;
                }
            }

            pivots.push_back(column);
            ++pivot_row;
        }

        if (pivot_row == rows) return true;
    }

    // Deterministic exact fallback for the rare case where every selected
    // full-rank minor vanishes modulo all trial primes.
    std::vector<std::vector<mpq_class>> exact(
        rows, std::vector<mpq_class>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            exact[i][j] = mpq_class(to_mpz_class(matrix[i][j]));
        }
    }

    pivots.clear();
    int pivot_row = 0;
    for (int column = 0; column < cols && pivot_row < rows; ++column) {
        int selected = pivot_row;
        while (selected < rows && exact[selected][column] == 0) ++selected;
        if (selected == rows) continue;
        if (selected != pivot_row) std::swap(exact[selected], exact[pivot_row]);

        const mpq_class pivot = exact[pivot_row][column];
        for (int j = column; j < cols; ++j) exact[pivot_row][j] /= pivot;
        for (int i = pivot_row + 1; i < rows; ++i) {
            if (exact[i][column] == 0) continue;
            const mpq_class factor = exact[i][column];
            for (int j = column; j < cols; ++j) {
                exact[i][j] -= factor * exact[pivot_row][j];
            }
        }
        pivots.push_back(column);
        ++pivot_row;
    }
    return pivot_row == rows;
}

bool recover_row_transform(const Matrix& original, const Matrix& candidate,
                           std::vector<std::vector<mpz_class>>& transform,
                           std::string* error) {
    const int dimension = original.get_rows();
    const int cols = original.get_cols();
    std::vector<int> pivot_columns;
    if (!find_independent_columns(original, pivot_columns)) {
        return fail(error, "the original block does not have full row rank");
    }

    // Solve A_selected^T * U^T = C_selected^T in one Gauss-Jordan pass.
    std::vector<std::vector<mpq_class>> augmented(
        dimension, std::vector<mpq_class>(2 * dimension));
    for (int row = 0; row < dimension; ++row) {
        for (int col = 0; col < dimension; ++col) {
            augmented[row][col] =
                mpq_class(to_mpz_class(original[col][pivot_columns[row]]));
            augmented[row][dimension + col] =
                mpq_class(to_mpz_class(candidate[col][pivot_columns[row]]));
        }
    }

    for (int column = 0; column < dimension; ++column) {
        int selected = column;
        while (selected < dimension && augmented[selected][column] == 0) ++selected;
        if (selected == dimension) {
            return fail(error, "selected exact column minor unexpectedly became singular");
        }
        if (selected != column) std::swap(augmented[selected], augmented[column]);

        const mpq_class pivot = augmented[column][column];
        for (int j = column; j < 2 * dimension; ++j) {
            augmented[column][j] /= pivot;
        }

        for (int i = 0; i < dimension; ++i) {
            if (i == column || augmented[i][column] == 0) continue;
            const mpq_class factor = augmented[i][column];
            for (int j = column; j < 2 * dimension; ++j) {
                augmented[i][j] -= factor * augmented[column][j];
            }
        }
    }

    transform.assign(dimension, std::vector<mpz_class>(dimension));
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            const mpq_class& value = augmented[j][dimension + i];
            if (value.get_den() != 1) {
                return fail(error,
                            "the sieve output is not an integral row transform of the MPZ block");
            }
            transform[i][j] = value.get_num();
        }
    }

    // Verify U * original == candidate over every column, not only the minor.
    for (int i = 0; i < dimension; ++i) {
        for (int column = 0; column < cols; ++column) {
            mpz_class sum = 0;
            for (int j = 0; j < dimension; ++j) {
                sum += transform[i][j] * to_mpz_class(original[j][column]);
            }
            if (sum != to_mpz_class(candidate[i][column])) {
                return fail(error,
                            "exact row-transform verification failed outside the pivot minor");
            }
        }
    }

    return true;
}

mpz_class bareiss_determinant(std::vector<std::vector<mpz_class>> matrix) {
    const int n = static_cast<int>(matrix.size());
    if (n == 0) return 1;
    if (n == 1) return matrix[0][0];

    mpz_class previous_pivot = 1;
    int sign = 1;

    for (int k = 0; k < n - 1; ++k) {
        int selected = k;
        while (selected < n && matrix[selected][k] == 0) ++selected;
        if (selected == n) return 0;
        if (selected != k) {
            std::swap(matrix[selected], matrix[k]);
            sign = -sign;
        }

        const mpz_class pivot = matrix[k][k];
        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                mpz_class numerator =
                    matrix[i][j] * pivot - matrix[i][k] * matrix[k][j];
                if (k > 0) {
                    mpz_divexact(numerator.get_mpz_t(), numerator.get_mpz_t(),
                                 previous_pivot.get_mpz_t());
                }
                matrix[i][j] = std::move(numerator);
            }
        }
        previous_pivot = pivot;
    }

    mpz_class determinant = matrix[n - 1][n - 1];
    if (sign < 0) determinant = -determinant;
    return determinant;
}

}  // namespace

Matrix parse_matrix(const std::string& text) {
    Matrix matrix;
    std::istringstream input(text);
    input >> matrix;
    if (!input && matrix.get_rows() == 0) {
        throw std::runtime_error("failed to parse an integer lattice matrix");
    }
    return matrix;
}

std::string dump_matrix(const Matrix& B) {
    std::ostringstream output;
    output << B;
    return output.str();
}

Matrix copy_block(const Matrix& B, int pos, int beta) {
    const int dimension = B.get_rows();
    const int cols = B.get_cols();
    const int actual_beta = std::max(0, std::min(beta, dimension - pos));
    Matrix out(actual_beta, cols);
    for (int i = 0; i < actual_beta; ++i) {
        for (int j = 0; j < cols; ++j) out[i][j] = B[pos + i][j];
    }
    return out;
}

bool write_block(Matrix& B, int pos, const Matrix& block, std::string* error) {
    if (pos < 0 || pos + block.get_rows() > B.get_rows() ||
        block.get_cols() != B.get_cols()) {
        return fail(error, "write_block shape or position mismatch");
    }
    for (int i = 0; i < block.get_rows(); ++i) {
        for (int j = 0; j < block.get_cols(); ++j) B[pos + i][j] = block[i][j];
    }
    return true;
}

bool matrices_equal(const Matrix& A, const Matrix& B) {
    if (A.get_rows() != B.get_rows() || A.get_cols() != B.get_cols()) return false;
    for (int i = 0; i < A.get_rows(); ++i) {
        for (int j = 0; j < A.get_cols(); ++j) {
            if (mpz_cmp(A[i][j].get_data(), B[i][j].get_data()) != 0) return false;
        }
    }
    return true;
}

void extract_scaled_matrix(const Matrix& B, std::vector<double>& M,
                           std::vector<double>& scales, int& n, int& cols) {
    n = B.get_rows();
    cols = B.get_cols();
    M.assign(static_cast<std::size_t>(n) * cols, 0.0);
    scales.assign(n, 0.0);
    std::vector<double> logs(cols), mantissas(cols);

    for (int i = 0; i < n; ++i) {
        double max_log = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < cols; ++j) {
            long exponent = 0;
            const double mantissa = mpz_get_d_2exp(&exponent, B[i][j].get_data());
            if (mantissa != 0.0) {
                const double log_value =
                    std::log(std::fabs(mantissa)) + static_cast<double>(exponent) * kLog2;
                logs[j] = log_value;
                mantissas[j] = mantissa;
                max_log = std::max(max_log, log_value);
            } else {
                logs[j] = -std::numeric_limits<double>::infinity();
                mantissas[j] = 0.0;
            }
        }

        scales[i] = std::isfinite(max_log) ? max_log : 0.0;
        for (int j = 0; j < cols; ++j) {
            if (std::isfinite(logs[j])) {
                M[static_cast<std::size_t>(i) * cols + j] =
                    std::copysign(std::exp(logs[j] - max_log), mantissas[j]);
            }
        }
    }
}

void gso_log_norms(const Matrix& B, std::vector<double>& gs) {
    std::vector<double> matrix;
    std::vector<double> scales;
    int n = 0;
    int cols = 0;
    extract_scaled_matrix(B, matrix, scales, n, cols);

    gs.assign(n, 0.0);
    std::vector<double> orthogonal(static_cast<std::size_t>(n) * cols, 0.0);
    std::vector<double> norm_squared(n, 0.0);
    std::vector<double> current(cols, 0.0);

    for (int i = 0; i < n; ++i) {
        const double* source = &matrix[static_cast<std::size_t>(i) * cols];
        std::copy(source, source + cols, current.begin());

        // A second modified-Gram-Schmidt pass is inexpensive at these dimensions
        // and materially improves the transaction metric on ill-conditioned bases.
        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < i; ++j) {
                if (norm_squared[j] <= 1e-300) continue;
                const double* previous =
                    &orthogonal[static_cast<std::size_t>(j) * cols];
                double product = 0.0;
                for (int k = 0; k < cols; ++k) product += current[k] * previous[k];
                const double coefficient = product / norm_squared[j];
                for (int k = 0; k < cols; ++k) {
                    current[k] -= coefficient * previous[k];
                }
            }
        }

        double squared = 0.0;
        for (int k = 0; k < cols; ++k) {
            orthogonal[static_cast<std::size_t>(i) * cols + k] = current[k];
            squared += current[k] * current[k];
        }
        norm_squared[i] = squared;
        gs[i] = squared > 1e-300
                    ? 0.5 * std::log(squared) + scales[i]
                    : -690.0;
    }
}

double log_potential(const Matrix& B) {
    std::vector<double> gs;
    gso_log_norms(B, gs);
    double potential = 0.0;
    const int dimension = static_cast<int>(gs.size());
    for (int i = 0; i < dimension; ++i) {
        potential += static_cast<double>(dimension - i) * gs[i];
    }
    return potential;
}

double first_gso_log_norm(const Matrix& B) {
    std::vector<double> gs;
    gso_log_norms(B, gs);
    return gs.empty() ? 1e300 : gs.front();
}

bool insert_and_lll(Matrix& B,
                    const std::vector<fplll::Z_NR<mpz_t>>& coefficients,
                    double delta,
                    std::string* error) {
    const int dimension = B.get_rows();
    const int cols = B.get_cols();
    if (static_cast<int>(coefficients.size()) != dimension) {
        return fail(error, "shortest-vector coefficient dimension mismatch");
    }

    Matrix expanded(dimension + 1, cols);
    fplll::Z_NR<mpz_t> accumulator;
    fplll::Z_NR<mpz_t> product;
    for (int column = 0; column < cols; ++column) {
        accumulator = 0;
        for (int i = 0; i < dimension; ++i) {
            if (coefficients[i].sgn() == 0) continue;
            product.mul(B[i][column], coefficients[i]);
            accumulator.add(accumulator, product);
        }
        expanded[0][column] = accumulator;
    }
    for (int i = 0; i < dimension; ++i) {
        for (int column = 0; column < cols; ++column) {
            expanded[i + 1][column] = B[i][column];
        }
    }

    try {
        fplll::lll_reduction(expanded, delta);
    } catch (const std::exception& ex) {
        return fail(error, std::string("LLL after exact insertion failed: ") + ex.what());
    } catch (...) {
        return fail(error, "LLL after exact insertion failed with an unknown exception");
    }

    int output_row = 0;
    for (int i = 0; i <= dimension && output_row < dimension; ++i) {
        bool zero = true;
        for (int column = 0; column < cols; ++column) {
            if (expanded[i][column].sgn() != 0) {
                zero = false;
                break;
            }
        }
        if (!zero) {
            for (int column = 0; column < cols; ++column) {
                B[output_row][column] = expanded[i][column];
            }
            ++output_row;
        }
    }

    if (output_row != dimension) {
        return fail(error, "exact insertion did not restore the original block rank");
    }
    return true;
}

bool recover_exact_unimodular_basis(const Matrix& original,
                                    double** approximate_hi,
                                    double** approximate_lo,
                                    int rows,
                                    int cols,
                                    Matrix& recovered,
                                    std::string* error) {
    if (!approximate_hi || !approximate_lo || rows != original.get_rows() ||
        cols != original.get_cols()) {
        return fail(error, "invalid approximate sieve basis dimensions");
    }

    Matrix candidate(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mpz_class rounded;
            if (!round_double_double(approximate_hi[i][j], approximate_lo[i][j],
                                     rounded)) {
                return fail(error,
                            "quad sieve output is not sufficiently close to an integer basis");
            }
            assign_mpz(candidate[i][j], rounded);
        }
    }

    std::vector<std::vector<mpz_class>> transform;
    if (!recover_row_transform(original, candidate, transform, error)) return false;

    const mpz_class determinant = bareiss_determinant(transform);
    if (determinant != 1 && determinant != -1) {
        return fail(error,
                    "the recovered integer transform is not unimodular; sieve output rejected");
    }

    recovered = std::move(candidate);
    return true;
}

std::uint64_t matrix_fingerprint(const Matrix& B) {
    // FNV-1a over a stable textual representation. This is a cache/task identity,
    // not a cryptographic digest.
    const std::string text = dump_matrix(B);
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : text) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace lattice_backend
