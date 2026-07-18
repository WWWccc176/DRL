#pragma once
#include "types.hpp"
#include <cstdint>

namespace lattice_backend {
ReduceResult reduce_extreme(int64_t matrix_id, Matrix& B, int pos, int beta, bool bool_sieve);
}
