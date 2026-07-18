//#include "lattice_cuda.h"
//#include <cuda_runtime.h>
//#include <mutex>
//
//static std::mutex g_cuda_mu;
//static bool g_checked = false, g_avail = false;
//
//// 复用设备缓冲，避免每步 cudaMalloc/Free
//static double* d_M = nullptr; static size_t cap_M = 0;
//static double* d_G = nullptr; static size_t cap_G = 0;
//static float*  d_C = nullptr; static size_t cap_C = 0;
//
//bool cuda_is_available() {
//    std::lock_guard<std::mutex> lk(g_cuda_mu);
//    if (!g_checked) {
//        int cnt = 0;
//        cudaError_t e = cudaGetDeviceCount(&cnt);
//        g_avail = (e == cudaSuccess && cnt > 0);
//        g_checked = true;
//    }
//    return g_avail;
//}
//
//// 每个线程算一个 (i,j) 内积 (i>=j), 对称写回
//__global__ void gram_kernel(const double* __restrict__ M, int n, int cols,
//                            double* __restrict__ G) {
//    int i = blockIdx.y * blockDim.y + threadIdx.y;
//    int j = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i < n && j < n && i >= j) {
//        double s = 0.0;
//        const double* ri = M + (size_t)i * cols;
//        const double* rj = M + (size_t)j * cols;
//        for (int k = 0; k < cols; ++k) s += ri[k] * rj[k];
//        G[(size_t)i * n + j] = s;
//        G[(size_t)j * n + i] = s;
//    }
//}
//
//// 由 Gram 归一化得到严格下三角 |cos|
//__global__ void cosine_kernel(const double* __restrict__ G, int n,
//                              float* __restrict__ C) {
//    int i = blockIdx.y * blockDim.y + threadIdx.y;
//    int j = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i < n && j < n) {
//        if (i > j) {
//            double denom = sqrt(G[(size_t)i * n + i] * G[(size_t)j * n + j]) + 1e-20;
//            C[(size_t)i * n + j] = (float)fabs(G[(size_t)i * n + j] / denom);
//        } else {
//            C[(size_t)i * n + j] = 0.0f;
//        }
//    }
//}
//
//static bool ensure_cap(void** ptr, size_t* cap, size_t need) {
//    if (*cap >= need) return true;
//    if (*ptr) cudaFree(*ptr);
//    *ptr = nullptr; *cap = 0;
//    if (cudaMalloc(ptr, need) != cudaSuccess) return false;
//    *cap = need;
//    return true;
//}
//
//bool cuda_gram_cosine(const double* M, int n, int cols,
//                      double* G_out, float* cosL_out) {
//    if (n <= 0 || cols <= 0) return false;
//    if (!cuda_is_available()) return false;
//
//    std::lock_guard<std::mutex> lk(g_cuda_mu);
//    size_t szM = (size_t)n * cols * sizeof(double);
//    size_t szG = (size_t)n * n   * sizeof(double);
//    size_t szC = (size_t)n * n   * sizeof(float);
//
//    if (!ensure_cap((void**)&d_M, &cap_M, szM)) return false;
//    if (!ensure_cap((void**)&d_G, &cap_G, szG)) return false;
//    if (!ensure_cap((void**)&d_C, &cap_C, szC)) return false;
//
//    if (cudaMemcpy(d_M, M, szM, cudaMemcpyHostToDevice) != cudaSuccess) return false;
//
//    dim3 blk(16, 16);
//    dim3 grd((n + 15) / 16, (n + 15) / 16);
//    gram_kernel<<<grd, blk>>>(d_M, n, cols, d_G);
//    cosine_kernel<<<grd, blk>>>(d_G, n, d_C);
//
//    if (cudaDeviceSynchronize() != cudaSuccess) return false;
//    if (cudaMemcpy(G_out, d_G, szG, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
//    if (cudaMemcpy(cosL_out, d_C, szC, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
//    return true;
//}
#include "lattice_cuda.h"
#include <cuda_runtime.h>
#include <mutex>
#include <vector>
#include <cmath>

static std::mutex g_cuda_mu;

bool cuda_is_available() {
    std::lock_guard<std::mutex> lk(g_cuda_mu);
    int cnt = 0;
    cudaError_t e = cudaGetDeviceCount(&cnt);
    return (e == cudaSuccess && cnt > 0);
}

// ---------- device helpers ----------
__device__ double atomicMaxDouble(double* addr, double val) {
    unsigned long long* a = (unsigned long long*)addr;
    unsigned long long old = *a, assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        if (cur >= val) break;
        old = atomicCAS(a, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(*a);
}

// ---------- Gram + cosine ----------
__global__ void gram_kernel(const double* M, int n, int cols, double* G) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n || j > i) return;
    const double* ri = M + (size_t)i * cols;
    const double* rj = M + (size_t)j * cols;
    double s = 0.0;
    for (int k = 0; k < cols; ++k) s += ri[k] * rj[k];
    G[(size_t)i * n + j] = s;
    G[(size_t)j * n + i] = s;
}

__global__ void cosine_stats_kernel(const double* G, int n,
                                    float* cosL, double* stats) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n || j >= i) return;
    double gii = G[(size_t)i * n + i];
    double gjj = G[(size_t)j * n + j];
    double denom = sqrt(gii * gjj) + 1e-20;
    double c = fabs(G[(size_t)i * n + j] / denom);
    cosL[(size_t)i * n + j] = (float)c;
    atomicAdd(&stats[1], c);
    atomicMaxDouble(&stats[0], c);
}

bool cuda_gram_cosine(const double* M, int n, int cols,
                      double* G_out, float* cosL_out, double* stats_out) {
    std::lock_guard<std::mutex> lk(g_cuda_mu);
    int cnt = 0;
    if (cudaGetDeviceCount(&cnt) != cudaSuccess || cnt == 0) return false;
    if (n <= 0 || cols <= 0) return false;

    double *dM = nullptr, *dG = nullptr, *dStats = nullptr;
    float* dCos = nullptr;
    size_t szM = (size_t)n * cols * sizeof(double);
    size_t szG = (size_t)n * n * sizeof(double);
    size_t szC = (size_t)n * n * sizeof(float);
    if (cudaMalloc(&dM, szM) != cudaSuccess) return false;
    if (cudaMalloc(&dG, szG) != cudaSuccess) { cudaFree(dM); return false; }
    if (cudaMalloc(&dCos, szC) != cudaSuccess) { cudaFree(dM); cudaFree(dG); return false; }
    if (cudaMalloc(&dStats, 2 * sizeof(double)) != cudaSuccess) {
        cudaFree(dM); cudaFree(dG); cudaFree(dCos); return false;
    }
    cudaMemcpy(dM, M, szM, cudaMemcpyHostToDevice);
    cudaMemset(dCos, 0, szC);
    cudaMemset(dStats, 0, 2 * sizeof(double));

    dim3 blk(16, 16), grd((n + 15) / 16, (n + 15) / 16);
    gram_kernel<<<grd, blk>>>(dM, n, cols, dG);
    cosine_stats_kernel<<<grd, blk>>>(dG, n, dCos, dStats);
    cudaDeviceSynchronize();

    if (G_out)    cudaMemcpy(G_out,    dG,    szG, cudaMemcpyDeviceToHost);
    if (cosL_out) cudaMemcpy(cosL_out, dCos,  szC, cudaMemcpyDeviceToHost);
    if (stats_out)cudaMemcpy(stats_out,dStats,2*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dM); cudaFree(dG); cudaFree(dCos); cudaFree(dStats);
    return true;
}

// ---------- sieve ----------
__global__ void matmul_XG(const double* X, const double* G, double* Y, int N, int d) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || k >= d) return;
    double s = 0.0;
    for (int l = 0; l < d; ++l) s += X[(size_t)i * d + l] * G[(size_t)l * d + k];
    Y[(size_t)i * d + k] = s;
}
__global__ void matmul_IP(const double* X, const double* Y, double* IP, int N, int d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    double s = 0.0;
    for (int k = 0; k < d; ++k) s += X[(size_t)i * d + k] * Y[(size_t)j * d + k];
    IP[(size_t)i * N + j] = s;
}
__global__ void diag_kernel(const double* IP, double* norms, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    norms[i] = IP[(size_t)i * N + i];
}
__global__ void reduce_kernel(const double* IP, const double* norms,
                              const double* X, double* Xnew, int N, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double ni = norms[i];
    double bestDelta = 0.0; int bestj = -1; long bestt = 0;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        double nj = norms[j];
        if (nj < 1e-9) continue;
        double ip = IP[(size_t)i * N + j];
        long t = llround(ip / nj);
        if (t == 0) continue;
        double newn = ni - 2.0 * (double)t * ip + (double)t * (double)t * nj;
        double delta = ni - newn;
        if (delta > bestDelta + 1e-12) { bestDelta = delta; bestj = j; bestt = t; }
    }
    const double* xi = X + (size_t)i * d;
    double* yi = Xnew + (size_t)i * d;
    if (bestj >= 0) {
        const double* xj = X + (size_t)bestj * d;
        for (int k = 0; k < d; ++k) yi[k] = xi[k] - (double)bestt * xj[k];
    } else {
        for (int k = 0; k < d; ++k) yi[k] = xi[k];
    }
}

double cuda_sieve_shortest(const double* G, int d, const double* X_init, int N,
                           int rounds, double* best_x_out, double* X_out) {
    std::lock_guard<std::mutex> lk(g_cuda_mu);
    int cnt = 0;
    if (cudaGetDeviceCount(&cnt) != cudaSuccess || cnt == 0) return -1.0;
    if (d <= 0 || N <= 0) return -1.0;

    size_t szG = (size_t)d * d * sizeof(double);
    size_t szX = (size_t)N * d * sizeof(double);
    size_t szIP = (size_t)N * N * sizeof(double);
    size_t szNv = (size_t)N * sizeof(double);

    double *dG=nullptr,*dX=nullptr,*dXnew=nullptr,*dY=nullptr,*dIP=nullptr,*dN=nullptr;
    auto cleanup = [&](){ cudaFree(dG);cudaFree(dX);cudaFree(dXnew);cudaFree(dY);cudaFree(dIP);cudaFree(dN); };
    if (cudaMalloc(&dG,szG)!=cudaSuccess){cleanup();return -1.0;}
    if (cudaMalloc(&dX,szX)!=cudaSuccess){cleanup();return -1.0;}
    if (cudaMalloc(&dXnew,szX)!=cudaSuccess){cleanup();return -1.0;}
    if (cudaMalloc(&dY,szX)!=cudaSuccess){cleanup();return -1.0;}
    if (cudaMalloc(&dIP,szIP)!=cudaSuccess){cleanup();return -1.0;}
    if (cudaMalloc(&dN,szNv)!=cudaSuccess){cleanup();return -1.0;}

    cudaMemcpy(dG, G, szG, cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X_init, szX, cudaMemcpyHostToDevice);

    std::vector<double> hnorms(N);
    double best = 1e300;
    dim3 blk(16, 16);
    dim3 gXG((d + 15) / 16, (N + 15) / 16);
    dim3 gIP((N + 15) / 16, (N + 15) / 16);
    int tb = 256, gb = (N + tb - 1) / tb;

    for (int r = 0; r < rounds; ++r) {
        matmul_XG<<<gXG, blk>>>(dX, dG, dY, N, d);
        matmul_IP<<<gIP, blk>>>(dX, dY, dIP, N, d);
        diag_kernel<<<gb, tb>>>(dIP, dN, N);
        cudaMemcpy(hnorms.data(), dN, szNv, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        int bi = -1; double bn = 1e300;
        for (int i = 0; i < N; ++i) {
            double v = hnorms[i];
            if (v > 1e-6 && v < bn) { bn = v; bi = i; }
        }
        if (bi >= 0 && bn < best) {
            best = bn;
            cudaMemcpy(best_x_out, dX + (size_t)bi * d,
                       (size_t)d * sizeof(double), cudaMemcpyDeviceToHost);
        }
        reduce_kernel<<<gb, tb>>>(dIP, dN, dX, dXnew, N, d);
        std::swap(dX, dXnew);
    }
    cudaDeviceSynchronize();
    if (X_out) cudaMemcpy(X_out, dX, szX, cudaMemcpyDeviceToHost);
    for (int k = 0; k < d; ++k) best_x_out[k] = std::round(best_x_out[k]);

    cleanup();
    return best;
}
