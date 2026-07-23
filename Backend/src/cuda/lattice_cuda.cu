#include "lattice_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <mutex>

namespace {

std::mutex g_cuda_mutex;
bool g_availability_checked = false;
bool g_cuda_available = false;

struct DeviceBuffers {
    double* matrix = nullptr;
    double* gram = nullptr;
    float* cosine = nullptr;
    double* statistics = nullptr;
    std::size_t matrix_capacity = 0;
    std::size_t gram_capacity = 0;
    std::size_t cosine_capacity = 0;

    ~DeviceBuffers() {
        if (matrix) cudaFree(matrix);
        if (gram) cudaFree(gram);
        if (cosine) cudaFree(cosine);
        if (statistics) cudaFree(statistics);
    }

    static bool reserve(void** pointer, std::size_t& capacity,
                        std::size_t required) {
        if (capacity >= required) return true;
        if (*pointer) {
            cudaFree(*pointer);
            *pointer = nullptr;
            capacity = 0;
        }
        if (cudaMalloc(pointer, required) != cudaSuccess) return false;
        capacity = required;
        return true;
    }

    bool ensure(std::size_t matrix_bytes, std::size_t gram_bytes,
                std::size_t cosine_bytes) {
        if (!reserve(reinterpret_cast<void**>(&matrix), matrix_capacity,
                     matrix_bytes)) return false;
        if (!reserve(reinterpret_cast<void**>(&gram), gram_capacity,
                     gram_bytes)) return false;
        if (!reserve(reinterpret_cast<void**>(&cosine), cosine_capacity,
                     cosine_bytes)) return false;
        if (!statistics &&
            cudaMalloc(reinterpret_cast<void**>(&statistics),
                       2 * sizeof(double)) != cudaSuccess) {
            return false;
        }
        return true;
    }
};

DeviceBuffers g_buffers;

__device__ double atomic_max_double(double* address, double value) {
    auto* integer_address = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *integer_address;
    while (true) {
        const unsigned long long assumed = old;
        if (__longlong_as_double(assumed) >= value) break;
        old = atomicCAS(integer_address, assumed, __double_as_longlong(value));
        if (old == assumed) break;
    }
    return __longlong_as_double(old);
}

__global__ void gram_kernel(const double* matrix, int rows, int cols,
                            double* gram) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows || j >= rows || j > i) return;

    const double* row_i = matrix + static_cast<std::size_t>(i) * cols;
    const double* row_j = matrix + static_cast<std::size_t>(j) * cols;
    double product = 0.0;
    for (int k = 0; k < cols; ++k) product += row_i[k] * row_j[k];

    gram[static_cast<std::size_t>(i) * rows + j] = product;
    gram[static_cast<std::size_t>(j) * rows + i] = product;
}

__global__ void cosine_statistics_kernel(const double* gram, int rows,
                                          float* cosine,
                                          double* statistics) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows || j >= rows || j >= i) return;

    const double diagonal_i = gram[static_cast<std::size_t>(i) * rows + i];
    const double diagonal_j = gram[static_cast<std::size_t>(j) * rows + j];
    const double denominator = sqrt(diagonal_i * diagonal_j) + 1e-20;
    const double value = fabs(
        gram[static_cast<std::size_t>(i) * rows + j] / denominator);

    cosine[static_cast<std::size_t>(i) * rows + j] =
        static_cast<float>(value);
    atomicAdd(&statistics[1], value);
    atomic_max_double(&statistics[0], value);
}

bool check(cudaError_t status) {
    return status == cudaSuccess;
}

}  // namespace

bool cuda_is_available() {
    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    if (!g_availability_checked) {
        int count = 0;
        g_cuda_available =
            cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
        g_availability_checked = true;
    }
    return g_cuda_available;
}

bool cuda_gram_cosine(const double* matrix, int rows, int cols,
                      double* gram_out, float* cosine_out,
                      double* statistics_out) {
    if (!matrix || rows <= 0 || cols <= 0) return false;

    std::lock_guard<std::mutex> lock(g_cuda_mutex);
    if (!g_availability_checked) {
        int count = 0;
        g_cuda_available =
            cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
        g_availability_checked = true;
    }
    if (!g_cuda_available) return false;

    const std::size_t matrix_bytes =
        static_cast<std::size_t>(rows) * cols * sizeof(double);
    const std::size_t gram_bytes =
        static_cast<std::size_t>(rows) * rows * sizeof(double);
    const std::size_t cosine_bytes =
        static_cast<std::size_t>(rows) * rows * sizeof(float);

    if (!g_buffers.ensure(matrix_bytes, gram_bytes, cosine_bytes)) return false;
    if (!check(cudaMemcpy(g_buffers.matrix, matrix, matrix_bytes,
                          cudaMemcpyHostToDevice))) return false;
    if (!check(cudaMemset(g_buffers.cosine, 0, cosine_bytes))) return false;
    if (!check(cudaMemset(g_buffers.statistics, 0, 2 * sizeof(double)))) return false;

    const dim3 block(16, 16);
    const dim3 grid((rows + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);
    gram_kernel<<<grid, block>>>(g_buffers.matrix, rows, cols, g_buffers.gram);
    if (!check(cudaGetLastError())) return false;
    cosine_statistics_kernel<<<grid, block>>>(
        g_buffers.gram, rows, g_buffers.cosine, g_buffers.statistics);
    if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) return false;

    if (gram_out &&
        !check(cudaMemcpy(gram_out, g_buffers.gram, gram_bytes,
                          cudaMemcpyDeviceToHost))) return false;
    if (cosine_out &&
        !check(cudaMemcpy(cosine_out, g_buffers.cosine, cosine_bytes,
                          cudaMemcpyDeviceToHost))) return false;
    if (statistics_out &&
        !check(cudaMemcpy(statistics_out, g_buffers.statistics,
                          2 * sizeof(double), cudaMemcpyDeviceToHost))) return false;
    return true;
}
