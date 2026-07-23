#pragma once

#include "types.hpp"
#include <string>
#include <vector>

namespace lattice_backend {

Matrix parse_matrix(const std::string& s);
std::string dump_matrix(const Matrix& B);
Matrix copy_block(const Matrix& B, int pos, int beta);
bool write_block(Matrix& B, int pos, const Matrix& block, std::string* error = nullptr);
bool matrices_equal(const Matrix& A, const Matrix& B);

void extract_scaled_matrix(const Matrix& B, std::vector<double>& M,
                           std::vector<double>& scales, int& n, int& cols);
void gso_log_norms(const Matrix& B, std::vector<double>& gs);
double log_potential(const Matrix& B);
double first_gso_log_norm(const Matrix& B);

bool insert_and_lll(Matrix& B,
                    const std::vector<fplll::Z_NR<mpz_t>>& coefficients,
                    double delta = 0.999,
                    std::string* error = nullptr);

// Convert a double-double/quad approximation back to an exact integer basis,
// recover the exact row transformation against `original`, verify that the
// transformation is integral and unimodular, and only then return the basis.
// This is the exact boundary between the approximate BGJ engine and MPZ state.
bool recover_exact_unimodular_basis(const Matrix& original,
                                    double** approximate_hi,
                                    double** approximate_lo,
                                    int rows,
                                    int cols,
                                    Matrix& recovered,
                                    std::string* error = nullptr);

std::uint64_t matrix_fingerprint(const Matrix& B);

}  // namespace lattice_backend
