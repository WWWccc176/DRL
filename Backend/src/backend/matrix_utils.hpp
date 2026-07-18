#pragma once
#include "types.hpp"
#include <string>
#include <vector>

namespace lattice_backend {
Matrix parse_matrix(const std::string& s);
std::string dump_matrix(const Matrix& B);
Matrix copy_block(const Matrix& B, int pos, int beta);
void write_block(Matrix& B, int pos, const Matrix& block);
void extract_scaled_matrix(const Matrix& B, std::vector<double>& M,
                           std::vector<double>& scales, int& n, int& cols);
void gso_log_norms(const Matrix& B, std::vector<double>& gs);
double log_potential(const Matrix& B);
double first_gso_log_norm(const Matrix& B);
void insert_and_lll(Matrix& B, const std::vector<fplll::Z_NR<mpz_t>>& coeff,
                    double delta = 0.999);
}
