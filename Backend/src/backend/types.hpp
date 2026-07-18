#pragma once
#include <fplll.h>
#include <string>

namespace lattice_backend {
using Matrix = fplll::ZZ_mat<mpz_t>;

struct ReduceResult {
    std::string backend = "none";
    bool accepted = false;
    int actual_beta = 0;
    int sieve_dimension = 0;
    long long database_vectors = 0;
    double time_ms = 0.0;
};
}
