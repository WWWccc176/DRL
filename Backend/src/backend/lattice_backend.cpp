#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fplll.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "types.hpp"
#include "matrix_utils.hpp"
#include "bkz2_engine.hpp"
#include "extreme_reducer.hpp"
#include "sieve_bridge.hpp"
#include "../cuda/lattice_cuda.h"

namespace py=pybind11;
using lattice_backend::Matrix;

static std::unordered_map<int64_t,Matrix> g_pool;
static int64_t g_next_id=1;
static std::mutex g_pool_mutex;

static int64_t create_matrix(const std::string& s) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    const int64_t id=g_next_id++;
    g_pool[id]=lattice_backend::parse_matrix(s);
    return id;
}

static int64_t create_matrix_lll(const std::string& s) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    const int64_t id=g_next_id++;
    g_pool[id]=lattice_backend::parse_matrix(s);
    fplll::lll_reduction(g_pool[id],0.999);
    return id;
}

static std::string dump_matrix(int64_t id) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it=g_pool.find(id);
    return it==g_pool.end()?"":lattice_backend::dump_matrix(it->second);
}

static void free_matrix(int64_t id) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    g_pool.erase(id);
}

static int64_t clone_matrix(int64_t id) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it=g_pool.find(id);
    if (it==g_pool.end()) return -1;
    const int64_t nid=g_next_id++;
    g_pool[nid]=it->second;
    return nid;
}

static py::dict evaluate_matrix(int64_t id) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it=g_pool.find(id);
    if (it==g_pool.end()) throw std::runtime_error("invalid matrix id");

    std::vector<double> M,scales,gs;
    int n=0,cols=0;
    lattice_backend::extract_scaled_matrix(it->second,M,scales,n,cols);
    lattice_backend::gso_log_norms(it->second,gs);

    std::vector<double> G((size_t)n*n,0.0);
    std::vector<float> C((size_t)n*n,0.0f);
    double stats[2]={0.0,0.0};
    bool gpu=false;
#ifdef USE_CUDA
    gpu=cuda_gram_cosine(M.data(),n,cols,G.data(),C.data(),stats);
#endif
    if (!gpu) {
        for (int i=0;i<n;++i) for (int j=0;j<=i;++j) {
            double s=0.0;
            for (int k=0;k<cols;++k)
                s+=M[(size_t)i*cols+k]*M[(size_t)j*cols+k];
            G[(size_t)i*n+j]=G[(size_t)j*n+i]=s;
        }
        for (int i=0;i<n;++i) for (int j=0;j<i;++j) {
            const double den=std::sqrt(G[(size_t)i*n+i]*G[(size_t)j*n+j])+1e-20;
            const double c=std::fabs(G[(size_t)i*n+j]/den);
            C[(size_t)i*n+j]=(float)c;
            stats[0]=std::max(stats[0],c);
            stats[1]+=c;
        }
    }

    py::array_t<double> gs_arr({n});
    std::copy(gs.begin(),gs.end(),gs_arr.mutable_data());
    py::array_t<float> cos_arr({n,n});
    std::copy(C.begin(),C.end(),cos_arr.mutable_data());

    py::dict d;
    d["gs_log_norms"]=gs_arr;
    d["cos_matrix"]=cos_arr;
    d["max_cos"]=stats[0];
    d["mean_cos"]=(n>1?stats[1]/((double)n*(n-1)/2.0):0.0);
    return d;
}

static py::dict reduce_extreme_api(int64_t matrix_id,int pos,int beta,bool bool_sieve) {
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it=g_pool.find(matrix_id);
    if (it==g_pool.end()) throw std::runtime_error("invalid matrix id");

    auto r=lattice_backend::reduce_extreme(matrix_id,it->second,pos,beta,bool_sieve);
    py::dict d;
    d["backend"]=r.backend;
    d["accepted"]=r.accepted;
    d["actual_beta"]=r.actual_beta;
    d["sieve_dimension"]=r.sieve_dimension;
    d["database_vectors"]=r.database_vectors;
    d["time_ms"]=r.time_ms;
    return d;
}

// Exact A11 initialization route: LLL has already been run by create_matrix_lll().
// This performs one GLOBAL BKZ 2.0 tour with the requested block size.
static py::dict reduce_bkz2_global_api(int64_t matrix_id,int beta,int loops) {
    const auto t0=std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it=g_pool.find(matrix_id);
    if (it==g_pool.end()) throw std::runtime_error("invalid matrix id");

    const int actual_beta=std::max(2,std::min(beta,it->second.get_rows()));
    lattice_backend::run_bkz2(it->second,actual_beta,std::max(1,loops),1.0);

    py::dict d;
    d["backend"]="bkz2_global";
    d["accepted"]=true;
    d["actual_beta"]=actual_beta;
    d["time_ms"]=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-t0).count();
    return d;
}

// Exact A11 cycle tail: sieve the requested local block, write it back only if its
// potential does not worsen, then run full-basis LLL and return.
static py::dict reduce_sieve_block_api(int64_t matrix_id,int pos,int beta) {
    const auto t0=std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(g_pool_mutex);
    auto it=g_pool.find(matrix_id);
    if (it==g_pool.end()) throw std::runtime_error("invalid matrix id");

    Matrix& B=it->second;
    if (pos<0 || pos>=B.get_rows() || beta<2)
        throw std::runtime_error("invalid sieve block");

    const int actual_beta=std::min(beta,B.get_rows()-pos);
    Matrix block=lattice_backend::copy_block(B,pos,actual_beta);
    const double pot_before=lattice_backend::log_potential(block);

    auto s=lattice_backend::run_local_extreme_sieve(block,matrix_id,pos);
    fplll::lll_reduction(block,0.999);

    const double pot_after=lattice_backend::log_potential(block);
    const bool accepted=std::isfinite(pot_after) && pot_after<=pot_before+1e-10;
    if (accepted)
        lattice_backend::write_block(B,pos,block);

    // User-requested final whole-basis LLL, regardless of whether the sieve insert
    // was accepted.
    fplll::lll_reduction(B,0.999);

    py::dict d;
    d["backend"]="local_bgj_sieve_final";
    d["accepted"]=accepted;
    d["actual_beta"]=actual_beta;
    d["sieve_dimension"]=s.final_csd;
    d["database_vectors"]=s.vectors;
    d["time_ms"]=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-t0).count();
    return d;
}

static py::dict reduce_compat(int64_t matrix_id,const std::string& method,int param,int pos) {
    const bool sieve=(method=="ORACLE" || method=="SIEVE" || method=="EXTREME");
    return reduce_extreme_api(matrix_id,pos,param,sieve);
}

PYBIND11_MODULE(my_project_backend,m) {
    m.doc()="Local extreme lattice backend: BKZ2.0 + exact enumeration + integrated BGJ/DH CUDA sieve";
    m.def("create_matrix",&create_matrix,py::arg("matrix_str"));
    m.def("create_matrix_lll",&create_matrix_lll,py::arg("matrix_str"));
    m.def("reduce_extreme",&reduce_extreme_api,
          py::arg("matrix_id"),py::arg("pos"),py::arg("beta"),py::arg("bool_sieve"));
    m.def("reduce_bkz2_global",&reduce_bkz2_global_api,
          py::arg("matrix_id"),py::arg("beta"),py::arg("loops")=1);
    m.def("reduce_sieve_block",&reduce_sieve_block_api,
          py::arg("matrix_id"),py::arg("pos"),py::arg("beta"));
    m.def("reduce",&reduce_compat,
          py::arg("matrix_id"),py::arg("method"),py::arg("param"),py::arg("pos"));
    m.def("evaluate_matrix",&evaluate_matrix,py::arg("matrix_id"));
    m.def("dump_matrix",&dump_matrix,py::arg("matrix_id"));
    m.def("free_matrix",&free_matrix,py::arg("matrix_id"));
    m.def("clone_matrix",&clone_matrix,py::arg("matrix_id"));
#ifdef USE_CUDA
    m.def("cuda_available",[](){return cuda_is_available();});
#else
    m.def("cuda_available",[](){return false;});
#endif
}
