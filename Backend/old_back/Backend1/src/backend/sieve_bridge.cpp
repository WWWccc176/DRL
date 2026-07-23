#include "sieve_bridge.hpp"
#include "matrix_utils.hpp"
#include "../sieve_engine/include/pool_hd.h"
#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <thread>

namespace fs=std::filesystem;
namespace lattice_backend {

static NTL::ZZ mpz_to_zz(const mpz_t z) {
    char* raw=mpz_get_str(nullptr,10,z);
    NTL::ZZ out;
    std::stringstream ss(raw?raw:"0"); ss>>out;
    if (raw) {
        void (*freefunc)(void*,size_t)=nullptr;
        mp_get_memory_functions(nullptr,nullptr,&freefunc);
        freefunc(raw,std::strlen(raw)+1);
    }
    return out;
}
static void zz_to_mpz(const NTL::ZZ& z, mpz_t out) {
    std::ostringstream os; os<<z; mpz_set_str(out,os.str().c_str(),10);
}

class CwdGuard {
public:
    explicit CwdGuard(const fs::path& p):old_(fs::current_path()) {
        fs::create_directories(p); fs::current_path(p);
    }
    ~CwdGuard(){ try{fs::current_path(old_);}catch(...){} }
private: fs::path old_;
};

static int run_best_bgj(Pool_hd_t& pool) {
    if (pool.CSD<112) return pool.bgj1_Sieve_hd();
    if (pool.CSD<=143) return pool.bgj2_Sieve_hd();
    if (pool.CSD<168) return pool.bgj3l_Sieve_hd();
    return pool.bgj4_Sieve_hd();
}

static long target_db_size(int csd) {
    long double x=3.2L*std::pow(4.0L/3.0L,(long double)csd*0.5L)-5.0L;
    const char* env=std::getenv("LATTICE_SIEVE_MAX_VECTORS");
    long long cap=0;
    if (env && *env) cap=std::atoll(env);
    if (cap<=0) {
        const long long default_bytes=18LL<<30;
        cap=default_bytes/190LL;
    }
    if (x>(long double)cap) x=(long double)cap;
    if (x<csd+64) x=csd+64;
    return (long)x;
}

SieveRunInfo run_local_extreme_sieve(Matrix& block, int64_t matrix_id, int global_pos) {
    SieveRunInfo info;
    const int d=block.get_rows(), cols=block.get_cols();
    if (d<40 || d>176 || cols<=0) return info;

    NTL::Mat<NTL::ZZ> A; A.SetDims(d,cols);
    for (int i=0;i<d;++i) for (int j=0;j<cols;++j) A[i][j]=mpz_to_zz(block[i][j].get_data());
    Lattice_QP L(A);
    L.LLL_QP(0.999);

    fs::path base=std::getenv("LATTICE_SIEVE_WORKDIR") ? std::getenv("LATTICE_SIEVE_WORKDIR") : "/tmp/my_project_sieve";
    fs::path work=base/(std::to_string(matrix_id)+"_"+std::to_string(global_pos)+"_"+std::to_string(d));
    fs::create_directories(work);
    CwdGuard cwd(work);

    const int max_sieve_dim=d-1;
    const int start_sieve_dim=std::min(max_sieve_dim, std::max(39,std::min(60,max_sieve_dim)));
    Pool_hd_t pool(&L);
    pool.set_sieving_context(d-start_sieve_dim,d);
    pool.set_boost_depth(0);
    const unsigned hc=std::thread::hardware_concurrency();
    pool.set_num_threads((long)std::max(8u,std::min(32u,hc?hc:8u)));
    pool.sampling(target_db_size(pool.CSD));

    while (pool.CSD<=max_sieve_dim) {
        const int ret=run_best_bgj(pool);
        if (ret==-1 && pool.CSD<=100 && pool.check_dim_lose()==-1) break;
        if (pool.CSD<max_sieve_dim) pool.extend_left(); else break;
    }

    pool.down_sieve_flag=1;
    const double eta=1.05;
    long inserted_pos=-1;
    int irc=-1;
    if (pool.CSD>=60) {
        const char* dh=std::getenv("LATTICE_DISABLE_DH");
        if (!(dh && (*dh=='1'||*dh=='t'||*dh=='T')))
            irc=pool.dh_insert(0,eta,0.0,&inserted_pos,0.0);
    }
    if (irc!=0) irc=pool.insert(0,eta,&inserted_pos,1);

    if (pool.CSD>=144) run_best_bgj(pool);
    info.final_csd=pool.CSD;
    info.vectors=pool.pwc_manager->num_vec();
    info.changed=(irc==0 && inserted_pos>=0);

    if (d<90) L.LLL_DEEP_QP(0.999);
    L.LLL_QP(0.999);
    L.to_int();
    MAT_QP b=L.get_b();
    for (int i=0;i<d;++i) for (int j=0;j<cols;++j) {
        mpz_set_d(block[i][j].get_data(), std::round(b.hi[i][j]));
    }
    return info;
}
}
