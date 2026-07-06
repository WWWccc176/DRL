#pragma once
// GPU kernels for the lattice backend.
// All matrices are row-major double unless noted.

// Gram + |cosine| lower-triangle + cosine stats, computed together on GPU.
//   G_out   : n*n   (full Gram of M, = M M^T)
//   cosL_out: n*n   (lower triangle |cos| angles, rest 0)
//   stats_out: length 2 -> [0]=max|cos|, [1]=sum|cos| over i>j pairs (may be null)
bool cuda_gram_cosine(const double* M, int n, int cols,
                      double* G_out, float* cosL_out,
                      double* stats_out = nullptr);

// Heuristic GPU sieve/Gauss reducer on a block Gram G (d*d, scaled).
//   X_init : N*d coefficient database (row = integer combo of block rows)
//   returns the best (shortest) squared norm found (in G's scaled metric),
//           or a negative value on failure.
//   best_x_out : d integer coeffs of the best vector
//   X_out      : N*d final (most-reduced) database, for reuse (may be null)
double cuda_sieve_shortest(const double* G, int d,
                           const double* X_init, int N,
                           int rounds, double* best_x_out,
                           double* X_out = nullptr);

bool cuda_is_available();
