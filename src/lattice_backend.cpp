//#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>
//#include <pybind11/stl.h>
//
//#include <fplll/fplll.h>
//#include <fplll/bkz_param.h>
//
//#include <sstream>
//#include <cmath>
//#include <cstring>
//#include <cstdint>
//#include <cstdlib>
//#include <vector>
//#include <unordered_map>
//#include <mutex>
//#include <algorithm>
//#include <stdexcept>
//
//#include "lattice_cuda.h"
//
//namespace py = pybind11;
//using namespace fplll;
//using MyMatrix = ZZ_mat<mpz_t>;
//
//static const double LOG2 = 0.6931471805599453;
//
//// ==================== 矩阵池 ====================
//static std::unordered_map<int64_t, MyMatrix> g_pool;
//static int64_t g_next_id = 0;
//static std::mutex g_pool_mutex;
//
//// ==================== 基础工具 ====================
//static MyMatrix parse_matrix_core(const std::string& s) {
//    MyMatrix B;
//    std::stringstream ss(s);
//    ss >> B;
//    return B;
//}
//
//static std::string dump_matrix_core(const MyMatrix& B) {
//    std::stringstream ss;
//    ss << "[";
//    for (int i = 0; i < B.get_rows(); ++i) {
//        ss << "[";
//        for (int j = 0; j < B.get_cols(); ++j) {
//            ss << B[i][j];
//            if (j + 1 < B.get_cols()) ss << " ";
//        }
//        ss << "]";
//        if (i + 1 < B.get_rows()) ss << "\n";
//    }
//    ss << "]";
//    return ss.str();
//}
//
//static void do_reduction(MyMatrix& B, const std::string& method, int param, int pos) {
//    int d = B.get_rows();
//    if (method == "LLL") {
//        lll_reduction(B, 0.99);
//    } else if (method == "LOCAL_BKZ") {
//        int beta = param;
//        int actual_beta = std::min(beta, d - pos);
//        if (actual_beta >= 2) {
//            int cols = B.get_cols();
//            MyMatrix B_local(actual_beta, cols);
//            for (int i = 0; i < actual_beta; ++i)
//                for (int j = 0; j < cols; ++j)
//                    B_local[i][j] = B[pos + i][j];
//
//            lll_reduction(B_local, 0.99);
//
//            if (actual_beta >= 4) {
//                int internal_beta = std::min(actual_beta, 100);
//                std::vector<Strategy> strategies;
//                try { strategies = load_strategies_json(FPLLL_DEFAULT_STRATEGY); }
//                catch (...) {}
//                BKZParam bkz_param(internal_beta, strategies);
//                bkz_param.flags = BKZ_AUTO_ABORT | BKZ_GH_BND;
//                if (internal_beta <= 20)      { bkz_param.gh_factor = 1.1;  bkz_param.max_loops = 4;  }
//                else if (internal_beta <= 35) { bkz_param.gh_factor = 1.05; bkz_param.max_loops = 8;  }
//                else                          { bkz_param.gh_factor = 1.0;  bkz_param.max_loops = 16; }
//                bkz_reduction(&B_local, NULL, bkz_param);
//            }
//
//            for (int i = 0; i < actual_beta; ++i)
//                for (int j = 0; j < cols; ++j)
//                    B[pos + i][j] = B_local[i][j];
//        }
//    }
//}
//
//// 直接从 mpz 矩阵抽取「按行 max-abs 归一化」的浮点矩阵 + 每行 log 尺度
//// (等价于原 Rust extract_float_data / parse_and_scale, 数值更稳健)
//static void extract_scaled_matrix(const MyMatrix& B, std::vector<double>& M,
//                                  std::vector<double>& scales, int& n, int& cols) {
//    n = B.get_rows();
//    cols = B.get_cols();
//    M.assign((size_t)n * cols, 0.0);
//    scales.assign(n, 0.0);
//
//    std::vector<double> logs(cols), mant(cols);
//    for (int i = 0; i < n; ++i) {
//        double max_log = -1e300;
//        for (int j = 0; j < cols; ++j) {
//            long e = 0;
//            double m = mpz_get_d_2exp(&e, B[i][j].get_data());
//            if (m != 0.0) {
//                double lv = std::log(std::fabs(m)) + (double)e * LOG2;
//                logs[j] = lv; mant[j] = m;
//                if (lv > max_log) max_log = lv;
//            } else { logs[j] = -1e300; mant[j] = 0.0; }
//        }
//        scales[i] = (max_log > -1e299) ? max_log : 0.0;
//        double* row = &M[(size_t)i * cols];
//        for (int j = 0; j < cols; ++j) {
//            if (logs[j] > -1e299) {
//                double sign = mant[j] > 0 ? 1.0 : -1.0;
//                row[j] = sign * std::exp(logs[j] - max_log);
//            }
//        }
//    }
//}
//
//static bool use_cuda_runtime() {
//    static int cached = -1;
//    if (cached < 0) {
//        const char* e = std::getenv("LATTICE_DISABLE_CUDA");
//        cached = (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T')) ? 0 : 1;
//    }
//    return cached == 1;
//}
//
//// 余弦下三角 + log_prod + min_norm (尺度无关, 故 GPU/CPU 结果一致)
//static void compute_cos_and_norms(const std::vector<double>& M,
//                                  const std::vector<double>& scales,
//                                  int n, int cols, std::vector<float>& cosL,
//                                  double& log_prod, double& min_norm) {
//    std::vector<double> G((size_t)n * n, 0.0);
//    cosL.assign((size_t)n * n, 0.0f);
//
//    bool gpu = false;
//#ifdef USE_CUDA
//    if (use_cuda_runtime())
//        gpu = cuda_gram_cosine(M.data(), n, cols, G.data(), cosL.data());
//#endif
//    if (!gpu) {  // CPU 回退
//        for (int i = 0; i < n; ++i) {
//            const double* ri = &M[(size_t)i * cols];
//            for (int j = 0; j <= i; ++j) {
//                const double* rj = &M[(size_t)j * cols];
//                double s = 0.0;
//                for (int k = 0; k < cols; ++k) s += ri[k] * rj[k];
//                G[(size_t)i * n + j] = s;
//                G[(size_t)j * n + i] = s;
//            }
//        }
//        for (int i = 0; i < n; ++i)
//            for (int j = 0; j < i; ++j) {
//                double denom = std::sqrt(G[(size_t)i*n+i] * G[(size_t)j*n+j]) + 1e-20;
//                cosL[(size_t)i*n+j] = (float)std::fabs(G[(size_t)i*n+j] / denom);
//            }
//    }
//
//    log_prod = 0.0;
//    double minlog = 1e300;
//    for (int i = 0; i < n; ++i) {
//        double gii = G[(size_t)i * n + i];
//        double tln = (gii > 1e-300) ? (0.5 * std::log(gii) + scales[i]) : -690.0;
//        log_prod += tln;
//        if (tln < minlog) minlog = tln;
//    }
//    min_norm = std::exp(minlog);
//}
//
//// 经典 Gram-Schmidt -> gs_log_norms (与原 Rust compute_gram_schmidt 算法完全一致)
//static void compute_gs(const std::vector<double>& M, const std::vector<double>& scales,
//                       int n, int cols, std::vector<double>& gs) {
//    gs.assign(n, 0.0);
//    std::vector<double> bstar((size_t)n * cols, 0.0);
//    std::vector<double> bnorm2(n, 0.0);
//    std::vector<double> v(cols, 0.0);
//
//    for (int i = 0; i < n; ++i) {
//        const double* mi = &M[(size_t)i * cols];
//        for (int k = 0; k < cols; ++k) v[k] = mi[k];
//        for (int j = 0; j < i; ++j) {
//            double denom = bnorm2[j];
//            if (denom > 1e-300) {
//                const double* bj = &bstar[(size_t)j * cols];
//                double dot = 0.0;
//                for (int k = 0; k < cols; ++k) dot += v[k] * bj[k];
//                double mu = dot / denom;
//                for (int k = 0; k < cols; ++k) v[k] -= mu * bj[k];
//            }
//        }
//        double* bi = &bstar[(size_t)i * cols];
//        double ns = 0.0;
//        for (int k = 0; k < cols; ++k) { bi[k] = v[k]; ns += v[k] * v[k]; }
//        bnorm2[i] = ns;
//        gs[i] = (ns > 1e-300) ? (0.5 * std::log(ns) + scales[i]) : -690.0;
//    }
//}
//
//// ==================== Python API (无 _rust 后缀) ====================
//static int64_t create_matrix(const std::string& matrix_str) {
//    std::lock_guard<std::mutex> lk(g_pool_mutex);
//    int64_t id = g_next_id++;
//    g_pool[id] = parse_matrix_core(matrix_str);
//    return id;
//}
//
//static int64_t create_matrix_lll(const std::string& matrix_str) {
//    std::lock_guard<std::mutex> lk(g_pool_mutex);
//    int64_t id = g_next_id++;
//    g_pool[id] = parse_matrix_core(matrix_str);
//    lll_reduction(g_pool[id], 0.99);
//    return id;
//}
//
//static py::dict reduce_matrix(int64_t matrix_id, const std::string& method, int param, int pos) {
//    std::lock_guard<std::mutex> lk(g_pool_mutex);
//    auto it = g_pool.find(matrix_id);
//    if (it == g_pool.end()) throw std::runtime_error("reduce: invalid matrix id");
//
//    do_reduction(it->second, method, param, pos);
//
//    std::vector<double> M, scales; int n, cols;
//    extract_scaled_matrix(it->second, M, scales, n, cols);
//
//    std::vector<float> cosL;
//    double log_prod, min_norm;
//    compute_cos_and_norms(M, scales, n, cols, cosL, log_prod, min_norm);
//
//    py::array_t<float> cos(std::vector<py::ssize_t>{n, n});
//    std::memcpy(cos.mutable_data(), cosL.data(), sizeof(float) * (size_t)n * n);
//
//    py::dict d;
//    d["log_prod"]   = log_prod;
//    d["min_norm"]   = min_norm;
//    d["cos_matrix"] = cos;
//    return d;
//}
//
//static py::dict evaluate_matrix(int64_t matrix_id) {
//    auto it = g_pool.find(matrix_id);
//    if (it == g_pool.end()) throw std::runtime_error("evaluate_matrix: invalid matrix id");
//
//    std::vector<double> M, scales; int n, cols;
//    extract_scaled_matrix(it->second, M, scales, n, cols);
//
//    std::vector<double> gs;
//    compute_gs(M, scales, n, cols, gs);
//
//    py::array_t<double> arr(std::vector<py::ssize_t>{n});
//    std::memcpy(arr.mutable_data(), gs.data(), sizeof(double) * (size_t)n);
//
//    py::dict d;
//    d["gs_log_norms"] = arr;
//    return d;
//}
//
//static std::string dump_matrix(int64_t matrix_id) {
//    std::lock_guard<std::mutex> lk(g_pool_mutex);
//    auto it = g_pool.find(matrix_id);
//    if (it == g_pool.end()) return std::string("");
//    return dump_matrix_core(it->second);
//}
//
//static void free_matrix(int64_t matrix_id) {
//    std::lock_guard<std::mutex> lk(g_pool_mutex);
//    g_pool.erase(matrix_id);
//}
//
//static int64_t clone_matrix(int64_t matrix_id) {
//    std::lock_guard<std::mutex> lk(g_pool_mutex);
//    auto it = g_pool.find(matrix_id);
//    if (it == g_pool.end()) return -1;
//    int rows = it->second.get_rows();
//    int cols = it->second.get_cols();
//    MyMatrix C(rows, cols);
//    for (int i = 0; i < rows; ++i)
//        for (int j = 0; j < cols; ++j)
//            C[i][j] = it->second[i][j];
//    int64_t new_id = g_next_id++;
//    g_pool[new_id] = std::move(C);
//    return new_id;
//}
//
//PYBIND11_MODULE(my_project_backend, m) {
//    m.doc() = "Lattice reduction backend (C++/CUDA + fplll)";
//    m.def("create_matrix",     &create_matrix,     py::arg("matrix_str"));
//    m.def("create_matrix_lll", &create_matrix_lll, py::arg("matrix_str"));
//    m.def("reduce",            &reduce_matrix,
//          py::arg("matrix_id"), py::arg("method"), py::arg("param"), py::arg("pos"));
//    m.def("evaluate_matrix",   &evaluate_matrix,   py::arg("matrix_id"));
//    m.def("dump_matrix",       &dump_matrix,       py::arg("matrix_id"));
//    m.def("free_matrix",       &free_matrix,       py::arg("matrix_id"));
//    m.def("clone_matrix",      &clone_matrix,      py::arg("matrix_id"));
//#ifdef USE_CUDA
//    m.def("cuda_available", [](){ return cuda_is_available(); });
//#else
//    m.def("cuda_available", [](){ return false; });
//#endif
//}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <fplll.h>

#include <vector>
#include <string>
#include <sstream>
#include <mutex>
#include <unordered_map>
#include <random>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <chrono>
#include <algorithm>

#include "lattice_cuda.h"

using namespace fplll;
namespace py = pybind11;

typedef ZZ_mat<mpz_t> MyMatrix;

static const double LOG2 = 0.69314718055994530942;
static const int SIEVE_THRESHOLD = 40;
static const int SIEVE_MAX_N = 4096;
static const int SAFE_BKZ_MAX = 28;

// ==================== matrix pool ====================
static std::unordered_map<int64_t, MyMatrix> g_pool;
static int64_t g_next_id = 1;
static std::mutex g_pool_mutex;

// ==================== parsing / dump ====================
static MyMatrix parse_matrix_core(const std::string& s) {
    MyMatrix B;
    std::istringstream is(s);
    is >> B;
    return B;
}

static std::string dump_matrix_core(const MyMatrix& B) {
    std::ostringstream os;
    os << B;
    return os.str();
}

// ==================== runtime CUDA switch ====================
static inline bool use_cuda_runtime() {
    static int cached = -1;

    if (cached < 0) {
        const char* env = std::getenv("LATTICE_DISABLE_CUDA");

        cached = (
            env &&
            (
                env[0] == '1' ||
                env[0] == 't' ||
                env[0] == 'T'
            )
        ) ? 0 : 1;
    }

    return cached == 1;
}

// ==================== BKZ strategies ====================
static const int MAX_BKZ_BLOCK = 128;     // 后端支持的 blocksize 硬上限
static bool g_strat_from_json = false;
static std::string g_strat_path;

static std::vector<Strategy>& default_strategies() {
    static std::vector<Strategy> strat;
    static std::once_flag once;
    std::call_once(once, [](){
        // 1) 环境变量优先（fplll 只认编译期烤死的路径，路径失效时可用它救场）
        const char* envp = std::getenv("FPLLL_STRATEGIES_JSON");
        if (envp && *envp) {
            try { strat = load_strategies_json(envp); g_strat_path = envp; }
            catch (...) { strat.clear(); }
        }
        // 2) 回退 fplll 默认路径
        if (strat.empty()) {
            try {
                std::string p = strategy_full_path("default.json");
                if (!p.empty()) { strat = load_strategies_json(p); g_strat_path = p; }
            } catch (...) { strat.clear(); }
        }
        g_strat_from_json = !strat.empty();
        if (!g_strat_from_json)
            std::fprintf(stderr,
                "[backend][warn] BKZ default.json NOT loaded -> EmptyStrategy fallback "
                "(no pruning, large-beta enum will be VERY slow). "
                "Set FPLLL_STRATEGIES_JSON=/path/to/default.json\n");
        // 3) 关键修复：无条件补齐到 MAX_BKZ_BLOCK+1 项。
        //    fplll 的 BKZParam 收到空 vector 会“就地”把它填到当次 block_size 为止；
        //    本 static 被所有调用共享，下次更大的 block_size 触发
        //    strategies[bs] 越界读 -> SIGSEGV。补齐后 vector 永不为空、
        //    覆盖一切 bs<=MAX_BKZ_BLOCK，BKZParam 不会再改写它。
        for (long b = (long)strat.size(); b <= MAX_BKZ_BLOCK; ++b)
            strat.emplace_back(Strategy::EmptyStrategy(b));
    });
    return strat;
}

// 所有 BKZ 的唯一入口：封顶 + 标志位齐全（BKZ_MAX_LOOPS 不给则 max_loops 被忽略）
static void run_bkz(MyMatrix& L, int bs, int max_loops, double gh_factor) {
    bs = std::max(2, std::min(bs, MAX_BKZ_BLOCK));
    BKZParam par(bs, default_strategies());
    par.flags     = BKZ_AUTO_ABORT | BKZ_GH_BND | BKZ_MAX_LOOPS;
    par.max_loops = std::max(1, max_loops);
    par.gh_factor = gh_factor;
    try { bkz_reduction(&L, NULL, par); } catch (...) {}
}// ==================== reusable sieve database (SieveState) ====================
struct SieveState {
    int64_t mid = -1;
    int pos = 0, beta = 0, N = 0;
    std::vector<double> G, X;
};
static std::vector<SieveState> g_cache;    // guarded by g_pool_mutex

static SieveState* find_cache(int64_t mid, int pos, int beta) {
    for (auto& s : g_cache)
        if (s.mid==mid && s.pos==pos && s.beta==beta) return &s;
    return nullptr;
}
static void store_cache(int64_t mid, int pos, int beta, int N,
                        std::vector<double>&& G, std::vector<double>&& X) {
    SieveState* s = find_cache(mid, pos, beta);
    if (!s) { if (g_cache.size()>=128) g_cache.erase(g_cache.begin());
              g_cache.emplace_back(); s=&g_cache.back(); }
    s->mid=mid; s->pos=pos; s->beta=beta; s->N=N;
    s->G=std::move(G); s->X=std::move(X);
}
static void invalidate_all_cache(int64_t mid) {
    g_cache.erase(std::remove_if(g_cache.begin(), g_cache.end(),
        [&](const SieveState& s){ return s.mid==mid; }), g_cache.end());
}
static void invalidate_overlapping(int64_t mid, int lo, int hi) {
    g_cache.erase(std::remove_if(g_cache.begin(), g_cache.end(),
        [&](const SieveState& s){
            if (s.mid!=mid) return false;
            int a=s.pos, b=s.pos+s.beta;
            return (a<hi && lo<b);
        }), g_cache.end());
}
static bool gram_close(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size()!=b.size() || a.empty()) return false;
    double m=0.0;
    for (size_t i=0;i<a.size();++i){ double d=std::fabs(a[i]-b[i]); if(d>m)m=d; }
    return m < 1e-6;
}

// ==================== scaled arithmetic helpers ====================
static void extract_scaled_matrix(const MyMatrix& B, std::vector<double>& M,
                                  std::vector<double>& scales, int& n, int& cols) {
    n = B.get_rows(); cols = B.get_cols();
    M.assign((size_t)n*cols, 0.0);
    scales.assign(n, 0.0);
    std::vector<double> logs(cols), mant(cols);
    for (int i=0;i<n;++i){
        double max_log=-1e300;
        for (int j=0;j<cols;++j){
            long e=0; double m=mpz_get_d_2exp(&e, B[i][j].get_data());
            if (m!=0.0){ double lv=std::log(std::fabs(m))+(double)e*LOG2;
                         logs[j]=lv; mant[j]=m; if(lv>max_log)max_log=lv; }
            else { logs[j]=-1e300; mant[j]=0.0; }
        }
        scales[i]=(max_log>-1e299)?max_log:0.0;
        double* row=&M[(size_t)i*cols];
        for (int j=0;j<cols;++j)
            if (logs[j]>-1e299){ double s=mant[j]>0?1.0:-1.0;
                                 row[j]=s*std::exp(logs[j]-max_log); }
    }
}

static void compute_gs(const std::vector<double>& M, const std::vector<double>& scales,
                       int n, int cols, std::vector<double>& gs) {
    gs.assign(n, 0.0);
    std::vector<double> bstar((size_t)n*cols, 0.0), bnorm2(n,0.0), v(cols,0.0);
    for (int i=0;i<n;++i){
        const double* mi=&M[(size_t)i*cols];
        for (int k=0;k<cols;++k) v[k]=mi[k];
        for (int j=0;j<i;++j){
            double denom=bnorm2[j];
            if (denom>1e-300){
                const double* bj=&bstar[(size_t)j*cols];
                double dot=0.0; for(int k=0;k<cols;++k) dot+=v[k]*bj[k];
                double mu=dot/denom;
                for(int k=0;k<cols;++k) v[k]-=mu*bj[k];
            }
        }
        double* bi=&bstar[(size_t)i*cols]; double ns=0.0;
        for (int k=0;k<cols;++k){ bi[k]=v[k]; ns+=v[k]*v[k]; }
        bnorm2[i]=ns;
        gs[i]=(ns>1e-300)?(0.5*std::log(ns)+scales[i]):-690.0;   // log||b_i*||
    }
}

// GSO log-norms of a block + local logPot (1-(2): profile-based gates)
static void block_gso(const MyMatrix& Bb, std::vector<double>& gs) {
    std::vector<double> M, scales; int n, cols;
    extract_scaled_matrix(Bb, M, scales, n, cols);
    compute_gs(M, scales, n, cols, gs);
}
static double block_logpot(const std::vector<double>& gs) {
    double p=0.0; int d=(int)gs.size();
    for (int i=0;i<d;++i) p += (double)(d-i)*gs[i];   // ~ sum (d-i) log||b_i*||
    return p;
}

static void block_gram_scaled(const MyMatrix& Bb, int d, int cols,
                              std::vector<double>& G) {
    long emax=LONG_MIN;
    for (int i=0;i<d;++i) for (int j=0;j<cols;++j){
        long e; double m=mpz_get_d_2exp(&e, Bb[i][j].get_data());
        if (m!=0.0 && e>emax) emax=e;
    }
    if (emax==LONG_MIN) emax=0;
    std::vector<double> Mb((size_t)d*cols,0.0);
    for (int i=0;i<d;++i) for (int j=0;j<cols;++j){
        long e; double m=mpz_get_d_2exp(&e, Bb[i][j].get_data());
        Mb[(size_t)i*cols+j]=(m==0.0)?0.0:m*std::ldexp(1.0,(int)(e-emax));
    }
    G.assign((size_t)d*d,0.0);
    for (int i=0;i<d;++i) for (int j=0;j<=i;++j){
        const double* ri=&Mb[(size_t)i*cols]; const double* rj=&Mb[(size_t)j*cols];
        double s=0.0; for(int k=0;k<cols;++k) s+=ri[k]*rj[k];
        G[(size_t)i*d+j]=s; G[(size_t)j*d+i]=s;
    }
    double maxd=0.0; for(int i=0;i<d;++i) maxd=std::max(maxd,G[(size_t)i*d+i]);
    if (maxd>0.0) for(auto& g:G) g/=maxd;
}

// ==================== insertion (1-(5): big-mpz coefficients) ====================
static void insert_and_reduce(MyMatrix& Bb, int d, int cols,
                              const std::vector<Z_NR<mpz_t>>& coeff) {
    MyMatrix T(d+1, cols);
    Z_NR<mpz_t> acc, tmp;
    for (int j=0;j<cols;++j){
        acc = 0;
        for (int i=0;i<d;++i){
            if (coeff[i].sgn()==0) continue;
            tmp.mul(Bb[i][j], coeff[i]);     // mpz * mpz  (no long truncation)
            acc.add(acc, tmp);
        }
        T[0][j]=acc;
    }
    for (int i=0;i<d;++i) for (int j=0;j<cols;++j) T[i+1][j]=Bb[i][j];
    lll_reduction(T, 0.99);
    int r=0;
    for (int i=0;i<=d && r<d;++i){
        bool zero=true;
        for (int j=0;j<cols;++j) if (T[i][j].sgn()!=0){ zero=false; break; }
        if (zero) continue;
        for (int j=0;j<cols;++j) Bb[r][j]=T[i][j];
        ++r;
    }
}

// ==================== ENUM_INSERT_ONE (2-(1), 1-(4)) ====================
// LLL -> shortest_vector -> insert at front -> LLL -> drop zeros -> write back.
// Gate: first GSO norm ||b_0*|| must strictly decrease, else revert.
static bool enum_insert_one(MyMatrix& Bb, int d, int cols) {
    lll_reduction(Bb, 0.99);
    if (d < 2) return false;
    std::vector<double> gs_b; block_gso(Bb, gs_b);
    double b0_before = gs_b.empty()?1e300:gs_b[0];

    MyMatrix snap(d, cols);
    for (int i=0;i<d;++i) for (int j=0;j<cols;++j) snap[i][j]=Bb[i][j];

    std::vector<Z_NR<mpz_t>> sol;
    int st;
    try { st = shortest_vector(Bb, sol); } catch(...) { return false; }
    if (st!=RED_SUCCESS || (int)sol.size()!=d) return false;
    bool nz=false; for (int i=0;i<d;++i) if (sol[i].sgn()!=0){ nz=true; break; }
    if (!nz) return false;

    insert_and_reduce(Bb, d, cols, sol);      // big-mpz safe
    std::vector<double> gs_a; block_gso(Bb, gs_a);
    double b0_after = gs_a.empty()?1e300:gs_a[0];

    if (b0_after < b0_before - 1e-9) return true;         // accept
    for (int i=0;i<d;++i) for (int j=0;j<cols;++j) Bb[i][j]=snap[i][j];  // revert
    return false;
}

// ==================== ENUM_BLOCK_PROCESS (2-(2), 1-(3)) ====================
// LLL -> strong local BKZ (blocksize=beta, BKZ2.0 strategies, no auto-abort)
// on the WHOLE block -> LLL -> write whole block back.  Gate: local logPot.
static bool enum_block_process(MyMatrix& Bb, int d, int cols, int beta) {
    lll_reduction(Bb, 0.99);
    if (d < 2) return true;
    std::vector<double> gs_b; block_gso(Bb, gs_b);
    for (double g : gs_b) if (!std::isfinite(g)) return false;   // 异常 profile 不进枚举
    double pot_before = block_logpot(gs_b);

    MyMatrix snap(d, cols);
    for (int i=0;i<d;++i) for (int j=0;j<cols;++j) snap[i][j]=Bb[i][j];

    run_bkz(Bb, std::min(d, std::max(2, beta)), std::min(d, 12), 1.05);
    lll_reduction(Bb, 0.99);

    std::vector<double> gs_a; block_gso(Bb, gs_a);
    double pot_after = block_logpot(gs_a);
    if (pot_after <= pot_before + 1e-9) return true;
    for (int i=0;i<d;++i) for (int j=0;j<cols;++j) Bb[i][j]=snap[i][j];
    return false;
}

// ==================== GPU sieve path (legacy ORACLE, with DB reuse) ====================
static bool sieve_block(int64_t mid, int pos, MyMatrix& Bb,
                        int d, int cols, std::vector<Z_NR<mpz_t>>& coeff) {
    if (!use_cuda_runtime() || !cuda_is_available()) return false;
    std::vector<double> G; block_gram_scaled(Bb, d, cols, G);
    double g00 = G.empty()?0.0:G[0];

    int Nreq=(int)std::ceil(3.2*std::pow(1.3333333333, d/2.0));
    int N=std::min(std::max(Nreq, d+16), SIEVE_MAX_N);

    std::vector<double> X; int rounds; bool warm=false;
    SieveState* hit=find_cache(mid,pos,d);
    if (hit && hit->N==N && gram_close(hit->G,G)){ X=hit->X; rounds=std::max(4,d/4); warm=true; }
    if (!warm){
        X.assign((size_t)N*d,0.0);
        for (int i=0;i<d;++i) X[(size_t)i*d+i]=1.0;
        thread_local std::mt19937 rng(12345);
        std::uniform_int_distribution<int> idx(0,d-1), cd(-1,1);
        for (int i=d;i<N;++i){
            double* xi=&X[(size_t)i*d]; int terms=2+(int)(rng()%3);
            for (int t=0;t<terms;++t){ int col=idx(rng); int c=cd(rng); if(c==0)c=1; xi[col]+=(double)c; }
        }
        rounds=std::max(8,d);
    }
    std::vector<double> bestx(d,0.0), Xout((size_t)N*d);
    double bn=cuda_sieve_shortest(G.data(), d, X.data(), N, rounds, bestx.data(), Xout.data());
    if (bn<0.0) return false;
    store_cache(mid,pos,d,N,std::move(G),std::move(Xout));
    if (!(bn < g00*(1.0-1e-6))) return false;
    coeff.resize(d); bool nz=false;
    for (int i=0;i<d;++i){ long c=(long)std::llround(bestx[i]); coeff[i]=(long)c; if(c)nz=true; }
    return nz;
}

// ==================== reduction dispatch ====================
struct ReduceInfo {
    std::string backend="none", enum_mode="";
    bool accepted=false;
    int actual_beta=0;
    double time_ms=0.0;
};

static void do_reduction(int64_t mid, MyMatrix& B, const std::string& method,
                         int param, int pos, ReduceInfo& info) {
    auto t0 = std::chrono::steady_clock::now();
    int d = B.get_rows();
    if (d<=0) { return; }

    if (method=="LLL") {
        lll_reduction(B, 0.99);
        invalidate_all_cache(mid);
        info.backend="lll"; info.accepted=true;
        info.time_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
        return;
    }
    if (pos<0 || pos>=d) { return; }
    int cols=B.get_cols();
    int actual_beta=0; bool changed=false;

    if (method=="LOCAL_BKZ") {
        actual_beta=std::min(param, d-pos);
        if (actual_beta<2) return;
        MyMatrix L(actual_beta, cols);
        for (int i=0;i<actual_beta;++i) for(int j=0;j<cols;++j) L[i][j]=B[pos+i][j];
        lll_reduction(L, 0.99);
        if (actual_beta>=4){
            int ib = std::min(actual_beta, MAX_BKZ_BLOCK);
            double ghf; int loops;
            if (ib<=20)      { ghf=1.1;  loops=4;  }
            else if (ib<=35) { ghf=1.05; loops=8;  }
            else             { ghf=1.0;  loops=16; }
            run_bkz(L, ib, loops, ghf);
        }        
        for (int i=0;i<actual_beta;++i) for(int j=0;j<cols;++j) B[pos+i][j]=L[i][j];
        info.backend="local_bkz"; info.accepted=true; changed=true;

    } else if (method=="ORACLE_ENUM_ONE") {
        actual_beta=std::min(param, d-pos);
        if (actual_beta<2) return;
        MyMatrix L(actual_beta, cols);
        for (int i=0;i<actual_beta;++i) for(int j=0;j<cols;++j) L[i][j]=B[pos+i][j];
        bool acc=enum_insert_one(L, actual_beta, cols);
        for (int i=0;i<actual_beta;++i) for(int j=0;j<cols;++j) B[pos+i][j]=L[i][j];
        info.backend="enum"; info.enum_mode="insert_one"; info.accepted=acc; changed=true;

    } else if (method=="ORACLE_ENUM_BLOCK") {
        actual_beta=std::min(param, d-pos);
        if (actual_beta<2) return;
        MyMatrix L(actual_beta, cols);
        for (int i=0;i<actual_beta;++i) for(int j=0;j<cols;++j) L[i][j]=B[pos+i][j];
        bool acc=enum_block_process(L, actual_beta, cols, actual_beta);
        for (int i=0;i<actual_beta;++i) for(int j=0;j<cols;++j) B[pos+i][j]=L[i][j];
        info.backend="enum"; info.enum_mode="block_process"; info.accepted=acc; changed=true;

    } else if (method=="ORACLE") {   // legacy auto-route
        actual_beta=std::min(param, d-pos);
        if (actual_beta<2) return;
        MyMatrix L(actual_beta, cols);
        for (int i=0;i<actual_beta;++i) for(int j=0;j<cols;++j) L[i][j]=B[pos+i][j];
        if (actual_beta < SIEVE_THRESHOLD) {
            bool acc=enum_block_process(L, actual_beta, cols, actual_beta);
            info.backend="enum"; info.enum_mode="block_process"; info.accepted=acc;
        } else {
            lll_reduction(L, 0.99);
            std::vector<Z_NR<mpz_t>> coeff;
            bool acc=sieve_block(mid,pos,L,actual_beta,cols,coeff);
            if (acc) insert_and_reduce(L, actual_beta, cols, coeff);
            info.backend="sieve"; info.accepted=acc;
        }
        for (int i=0;i<actual_beta;++i) for(int j=0;j<cols;++j) B[pos+i][j]=L[i][j];
        changed=true;
    } else {
        return;
    }

    info.actual_beta=actual_beta;
    if (changed) invalidate_overlapping(mid, pos, pos+actual_beta);   // 2-(5)
    info.time_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
}

// ==================== stats (GPU-fused cos, §4) ====================
struct BasisStats {
    double log_prod=0.0, min_norm=0.0;
    double min_log_norm=0.0, max_log_norm=0.0, mean_log_norm=0.0;
    double max_cos=0.0, mean_cos=0.0;
};
static void compute_cos_and_stats(const std::vector<double>& M,
                                  const std::vector<double>& scales,
                                  int n, int cols, std::vector<float>& cosL,
                                  BasisStats& out) {
    std::vector<double> G((size_t)n*n,0.0);
    cosL.assign((size_t)n*n,0.0f);
    double statbuf[2]={0.0,0.0}; bool gpu=false;
#ifdef USE_CUDA
    if (use_cuda_runtime())
        gpu=cuda_gram_cosine(M.data(),n,cols,G.data(),cosL.data(),statbuf);
#endif
    if (!gpu){
        for (int i=0;i<n;++i){ const double* ri=&M[(size_t)i*cols];
            for (int j=0;j<=i;++j){ const double* rj=&M[(size_t)j*cols];
                double s=0.0; for(int k=0;k<cols;++k) s+=ri[k]*rj[k];
                G[(size_t)i*n+j]=s; G[(size_t)j*n+i]=s; } }
        double mc=0.0,sc=0.0;
        for (int i=0;i<n;++i) for(int j=0;j<i;++j){
            double den=std::sqrt(G[(size_t)i*n+i]*G[(size_t)j*n+j])+1e-20;
            double c=std::fabs(G[(size_t)i*n+j]/den);
            cosL[(size_t)i*n+j]=(float)c; sc+=c; if(c>mc)mc=c;
        }
        statbuf[0]=mc; statbuf[1]=sc;
    }
    double cnt=(double)n*(n-1)/2.0;
    out.max_cos=statbuf[0]; out.mean_cos=(cnt>0)?statbuf[1]/cnt:0.0;
    out.log_prod=0.0;
    double minl=1e300,maxl=-1e300,suml=0.0;
    for (int i=0;i<n;++i){
        double gii=G[(size_t)i*n+i];
        double tln=(gii>1e-300)?(0.5*std::log(gii)+scales[i]):-690.0;
        out.log_prod+=tln; suml+=tln; if(tln<minl)minl=tln; if(tln>maxl)maxl=tln;
    }
    out.min_log_norm=minl; out.max_log_norm=maxl;
    out.mean_log_norm=suml/std::max(1,n); out.min_norm=std::exp(minl);
}

// ==================== Python API ====================
static int64_t create_matrix(const std::string& s){
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    int64_t id=g_next_id++; g_pool[id]=parse_matrix_core(s); return id;
}
static int64_t create_matrix_lll(const std::string& s){
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    int64_t id=g_next_id++; g_pool[id]=parse_matrix_core(s);
    lll_reduction(g_pool[id],0.99); return id;
}

static py::dict reduce_matrix(int64_t matrix_id, const std::string& method,
                              int param, int pos){
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it=g_pool.find(matrix_id);
    if (it==g_pool.end()) throw std::runtime_error("reduce: invalid matrix id");

    ReduceInfo info;
    do_reduction(matrix_id, it->second, method, param, pos, info);

    std::vector<double> M, scales; int n, cols;
    extract_scaled_matrix(it->second, M, scales, n, cols);
    std::vector<float> cosL; BasisStats st;
    compute_cos_and_stats(M, scales, n, cols, cosL, st);

    py::array_t<float> cos(std::vector<py::ssize_t>{n,n});
    std::memcpy(cos.mutable_data(), cosL.data(), sizeof(float)*(size_t)n*n);

    py::dict d;
    d["log_prod"]=st.log_prod; d["min_norm"]=st.min_norm;
    d["min_log_norm"]=st.min_log_norm; d["max_log_norm"]=st.max_log_norm;
    d["mean_log_norm"]=st.mean_log_norm;
    d["max_cos"]=st.max_cos; d["mean_cos"]=st.mean_cos;
    d["cos_matrix"]=cos;
    // 2-(6): provenance for later comparison with LOCAL_BKZ / G6K
    d["backend"]=info.backend; d["enum_mode"]=info.enum_mode;
    d["accepted"]=info.accepted; d["actual_beta"]=info.actual_beta;
    d["time_ms"]=info.time_ms;
    return d;
}

static py::dict evaluate_matrix(int64_t matrix_id){
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it=g_pool.find(matrix_id);
    if (it==g_pool.end()) throw std::runtime_error("evaluate_matrix: invalid matrix id");
    std::vector<double> M, scales; int n, cols;
    extract_scaled_matrix(it->second, M, scales, n, cols);
    std::vector<double> gs; compute_gs(M, scales, n, cols, gs);
    py::array_t<double> arr(std::vector<py::ssize_t>{n});
    std::memcpy(arr.mutable_data(), gs.data(), sizeof(double)*(size_t)n);
    py::dict d; d["gs_log_norms"]=arr; return d;
}

static std::string dump_matrix(int64_t matrix_id){
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it=g_pool.find(matrix_id); if(it==g_pool.end()) return "";
    return dump_matrix_core(it->second);
}
static void free_matrix(int64_t matrix_id){
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    g_pool.erase(matrix_id); invalidate_all_cache(matrix_id);
}
static int64_t clone_matrix(int64_t matrix_id){
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it=g_pool.find(matrix_id); if(it==g_pool.end()) return -1;
    int r=it->second.get_rows(), c=it->second.get_cols();
    MyMatrix C(r,c); for(int i=0;i<r;++i) for(int j=0;j<c;++j) C[i][j]=it->second[i][j];
    int64_t nid=g_next_id++; g_pool[nid]=std::move(C); return nid;   // no cache copy
}

// ---- 3-(1): G6K bridge ----
static std::string dump_block(int64_t matrix_id, int pos, int beta){
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it=g_pool.find(matrix_id); if(it==g_pool.end()) return "";
    int d=it->second.get_rows(), cols=it->second.get_cols();
    if (pos<0||pos>=d) return "";
    int ab=std::min(beta, d-pos); if(ab<1) return "";
    MyMatrix Bb(ab, cols);
    for (int i=0;i<ab;++i) for(int j=0;j<cols;++j) Bb[i][j]=it->second[pos+i][j];
    return dump_matrix_core(Bb);
}

// coeffs are decimal strings, w.r.t. the ORIGINAL block basis rows (length = actual_beta).
static py::dict insert_coeff_vector(int64_t matrix_id, int pos, int beta,
                                    const std::vector<std::string>& coeffs){
    auto t0=std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    py::dict out; out["accepted"]=false; out["actual_beta"]=0;
    auto it=g_pool.find(matrix_id);
    if (it==g_pool.end()) throw std::runtime_error("insert_coeff_vector: invalid matrix id");
    MyMatrix& B=it->second;
    int d=B.get_rows(), cols=B.get_cols();
    if (pos<0||pos>=d) return out;
    int ab=std::min(beta, d-pos);
    if (ab<2 || (int)coeffs.size()!=ab) return out;

    MyMatrix L(ab, cols);
    for (int i=0;i<ab;++i) for(int j=0;j<cols;++j) L[i][j]=B[pos+i][j];

    std::vector<double> gs_b; block_gso(L, gs_b);
    double b0_before=gs_b.empty()?1e300:gs_b[0];

    std::vector<Z_NR<mpz_t>> c(ab);
    for (int i=0;i<ab;++i){
        // Z_NR<mpz_t> 的默认构造函数已经对存储区域执行了 mpz_init；
        // 将十进制字符串直接解析进去（没有截断，没有临时变量）。
        if (mpz_set_str(c[i].get_data(), coeffs[i].c_str(), 10) != 0)
            return out;   // 无效的十进制字符串 -> 拒绝
    }    
    bool nz=false; for(int i=0;i<ab;++i) if(c[i].sgn()!=0){ nz=true; break; }
    if (!nz) return out;

    MyMatrix snap(ab, cols);
    for (int i=0;i<ab;++i) for(int j=0;j<cols;++j) snap[i][j]=L[i][j];

    insert_and_reduce(L, ab, cols, c);
    std::vector<double> gs_a; block_gso(L, gs_a);
    double b0_after=gs_a.empty()?1e300:gs_a[0];

    bool accepted = (b0_after < b0_before - 1e-9);
    if (accepted){
        for (int i=0;i<ab;++i) for(int j=0;j<cols;++j) B[pos+i][j]=L[i][j];
        invalidate_overlapping(matrix_id, pos, pos+ab);
    }
    out["accepted"]=accepted; out["actual_beta"]=ab;
    out["time_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    return out;
}

PYBIND11_MODULE(my_project_backend, m){
    m.doc()="Lattice backend (fplll + CUDA): BKZ2.0, enum-oracle modes, sieve DB, G6K bridge";
    m.def("create_matrix",     &create_matrix,     py::arg("matrix_str"));
    m.def("create_matrix_lll", &create_matrix_lll, py::arg("matrix_str"));
    m.def("reduce",            &reduce_matrix,
          py::arg("matrix_id"), py::arg("method"), py::arg("param"), py::arg("pos"));
    m.def("evaluate_matrix",   &evaluate_matrix,   py::arg("matrix_id"));
    m.def("dump_matrix",       &dump_matrix,       py::arg("matrix_id"));
    m.def("free_matrix",       &free_matrix,       py::arg("matrix_id"));
    m.def("clone_matrix",      &clone_matrix,      py::arg("matrix_id"));
    m.def("dump_block",        &dump_block,
          py::arg("matrix_id"), py::arg("pos"), py::arg("beta"));
    m.def("insert_coeff_vector", &insert_coeff_vector,
          py::arg("matrix_id"), py::arg("pos"), py::arg("beta"), py::arg("coeffs"));
    m.def("strategies_info", [](){
        auto& s = default_strategies();          // 触发加载
        py::dict d;
        d["from_json"] = g_strat_from_json;
        d["path"]      = g_strat_path;
        d["count"]     = (int)s.size();          // 应恒为 MAX_BKZ_BLOCK+1 = 129
        return d;
    });
#ifdef USE_CUDA
    m.def("cuda_available", [](){ return cuda_is_available(); });
#else
    m.def("cuda_available", [](){ return false; });
#endif
}
