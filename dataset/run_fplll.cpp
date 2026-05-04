#include <fplll/fplll.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

using namespace std;
using namespace fplll;
using MyMatrix = ZZ_mat<mpz_t>;

// ==================== 内部工具 ====================
static MyMatrix parse_matrix_core(const std::string& input_str) {
    MyMatrix B;
    std::stringstream ss(input_str);
    ss >> B;
    return B;
}

static std::string dump_matrix_core(const MyMatrix& B) {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < B.get_rows(); ++i) {
        ss << "[";
        for (int j = 0; j < B.get_cols(); ++j) {
            ss << B[i][j];
            if (j + 1 < B.get_cols()) ss << " ";
        }
        ss << "]";
        if (i + 1 < B.get_rows()) ss << "\n";
    }
    ss << "]";
    return ss.str();
}

// ==================== 主函数 ====================
int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc != 2) {
        cerr << "[-] 用法: " << argv[0] << " <矩阵文件路径.txt>" << endl;
        cerr << "[-] 示例: " << argv[0] << " svpchallengedim67seed13.txt" << endl;
        return 1;
    }

    string filename = argv[1];
    cout << "[*] 正在尝试读取文件: " << filename << " ..." << endl;

    // 读取文件
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "[-] 错误: 无法打开文件! 请确认文件路径是否正确。" << endl;
        return 1;
    }

    stringstream buffer;
    buffer << file.rdbuf();
    string file_content = buffer.str();
    file.close();

    cout << "[*] 文件读取成功，正在解析矩阵..." << endl;
    
    // 解析矩阵 (fplll 支持直接从包含 [[]] 的文本流中解析)
    MyMatrix B = parse_matrix_core(file_content);

    if (B.get_rows() == 0 || B.get_cols() == 0) {
        cerr << "[-] 错误: 矩阵为空或解析失败！请检查文件内容格式是否为标准的 [[...], [...]]" << endl;
        return 1;
    }

    cout << "[+] 矩阵加载成功. 维度: " << B.get_rows() << " x " << B.get_cols() << endl;

    // 执行 LLL 约化
    cout << "[*] 正在执行 LLL 约化 (delta = 0.99)..." << endl;
    
    int status = lll_reduction(B, 0.99); 

    if (status != RED_SUCCESS) {
        cerr << "[-] LLL 约化失败，状态码: " << status << endl;
        return 1;
    }

    cout << "[+] LLL 约化完成。" << endl;

    // 打印最短向量 (第一行) 的前 10 个元素作为验证
    cout << "[+] 约化后的最短向量 v_1 (部分预览):" << endl;
    cout << "[ ";
    for (int j = 0; j < min(10, B.get_cols()); ++j) {
        cout << B[0][j] << " ";
    }
    if (B.get_cols() > 10) cout << "... ";
    cout << "]" << endl;

    string out_filename = filename + "lll_reduced.txt";
    ofstream out_file(out_filename);
    if (out_file.is_open()) {
        out_file << dump_matrix_core(B);
        out_file.close();
        cout << "[+] 约化后的矩阵已保存至: " << out_filename << endl;
    }
    
    return 0;
}
