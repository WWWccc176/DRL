#include "matrix_utils.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace lattice_backend {
static constexpr double LOG2C = 0.69314718055994530942;

Matrix parse_matrix(const std::string& s) {
    Matrix B; std::istringstream is(s); is >> B; return B;
}
std::string dump_matrix(const Matrix& B) {
    std::ostringstream os; os << B; return os.str();
}
Matrix copy_block(const Matrix& B, int pos, int beta) {
    const int d = B.get_rows(), cols = B.get_cols();
    const int ab = std::max(0, std::min(beta, d - pos));
    Matrix out(ab, cols);
    for (int i=0;i<ab;++i) for (int j=0;j<cols;++j) out[i][j]=B[pos+i][j];
    return out;
}
void write_block(Matrix& B, int pos, const Matrix& block) {
    for (int i=0;i<block.get_rows();++i)
        for (int j=0;j<block.get_cols();++j) B[pos+i][j]=block[i][j];
}
void extract_scaled_matrix(const Matrix& B, std::vector<double>& M,
                           std::vector<double>& scales, int& n, int& cols) {
    n=B.get_rows(); cols=B.get_cols();
    M.assign((size_t)n*cols,0.0); scales.assign(n,0.0);
    std::vector<double> logs(cols), mant(cols);
    for (int i=0;i<n;++i) {
        double max_log=-1e300;
        for (int j=0;j<cols;++j) {
            long e=0; const double m=mpz_get_d_2exp(&e,B[i][j].get_data());
            if (m!=0.0) {
                const double lv=std::log(std::fabs(m))+(double)e*LOG2C;
                logs[j]=lv; mant[j]=m; max_log=std::max(max_log,lv);
            } else { logs[j]=-1e300; mant[j]=0.0; }
        }
        scales[i]=(max_log>-1e299)?max_log:0.0;
        for (int j=0;j<cols;++j) if (logs[j]>-1e299)
            M[(size_t)i*cols+j]=(mant[j]>0?1.0:-1.0)*std::exp(logs[j]-max_log);
    }
}
void gso_log_norms(const Matrix& B, std::vector<double>& gs) {
    std::vector<double> M, scales; int n=0, cols=0;
    extract_scaled_matrix(B,M,scales,n,cols);
    gs.assign(n,0.0);
    std::vector<double> bstar((size_t)n*cols,0.0), bnorm2(n,0.0), v(cols,0.0);
    for (int i=0;i<n;++i) {
        const double* mi=&M[(size_t)i*cols];
        std::copy(mi,mi+cols,v.begin());
        for (int j=0;j<i;++j) if (bnorm2[j]>1e-300) {
            const double* bj=&bstar[(size_t)j*cols];
            double dot=0.0; for (int k=0;k<cols;++k) dot+=v[k]*bj[k];
            const double mu=dot/bnorm2[j];
            for (int k=0;k<cols;++k) v[k]-=mu*bj[k];
        }
        double ns=0.0;
        for (int k=0;k<cols;++k) { bstar[(size_t)i*cols+k]=v[k]; ns+=v[k]*v[k]; }
        bnorm2[i]=ns;
        gs[i]=(ns>1e-300)?0.5*std::log(ns)+scales[i]:-690.0;
    }
}
double log_potential(const Matrix& B) {
    std::vector<double> gs; gso_log_norms(B,gs);
    double p=0.0; const int d=(int)gs.size();
    for (int i=0;i<d;++i) p+=(double)(d-i)*gs[i];
    return p;
}
double first_gso_log_norm(const Matrix& B) {
    std::vector<double> gs; gso_log_norms(B,gs); return gs.empty()?1e300:gs[0];
}
void insert_and_lll(Matrix& B, const std::vector<fplll::Z_NR<mpz_t>>& coeff, double delta) {
    const int d=B.get_rows(), cols=B.get_cols();
    if ((int)coeff.size()!=d) return;
    Matrix T(d+1,cols);
    fplll::Z_NR<mpz_t> acc,tmp;
    for (int j=0;j<cols;++j) {
        acc=0;
        for (int i=0;i<d;++i) if (coeff[i].sgn()!=0) {
            tmp.mul(B[i][j],coeff[i]); acc.add(acc,tmp);
        }
        T[0][j]=acc;
    }
    for (int i=0;i<d;++i) for (int j=0;j<cols;++j) T[i+1][j]=B[i][j];
    fplll::lll_reduction(T,delta);
    int r=0;
    for (int i=0;i<=d && r<d;++i) {
        bool zero=true; for (int j=0;j<cols;++j) if (T[i][j].sgn()!=0) {zero=false;break;}
        if (!zero) { for (int j=0;j<cols;++j) B[r][j]=T[i][j]; ++r; }
    }
}
}
