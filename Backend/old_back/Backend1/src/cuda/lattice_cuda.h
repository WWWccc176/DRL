#pragma once

bool cuda_gram_cosine(const double* M, int n, int cols,
                      double* G_out, float* cosL_out,
                      double* stats_out = nullptr);

bool cuda_is_available();
