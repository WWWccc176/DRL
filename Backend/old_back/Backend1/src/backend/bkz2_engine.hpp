#pragma once
#include "types.hpp"

namespace lattice_backend {
void run_bkz2(Matrix& B, int block_size, int loops, double gh_factor=1.0);
void run_extreme_bkz2_preconditioner(Matrix& B);
}
