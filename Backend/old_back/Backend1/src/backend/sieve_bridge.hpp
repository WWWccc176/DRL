#pragma once
#include "types.hpp"
#include <cstdint>

namespace lattice_backend {
struct SieveRunInfo {
    bool changed=false;
    int final_csd=0;
    long long vectors=0;
};
SieveRunInfo run_local_extreme_sieve(Matrix& block, int64_t matrix_id, int global_pos);
}
