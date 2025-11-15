#include <chrono>
#include <vector>
#include <iostream>

using namespace std;

void vec_add_host(const float* A, const float* B, float* C, int N);
void vec_add_device(const float* A, const float* B, float* C, int N);


void createLargeDataset(std::vector<float>& a, std::vector<float>& b, size_t N) {
    a.resize(N);
    b.resize(N);

    for(size_t i = 0; i < N; i++) {
        a[i] = static_cast<float>(i%1000);
        b[i] = static_cast<float>(i%1000);
    }
}

int main() {
    size_t N = 1000000000;

    // A and B
    std::vector<float> h_A(N), h_B(N);

    // Host and Device res
    std::vector<float> h_C_device(N), h_C_host(N);
    h_C_device.resize(N);
    h_C_host.resize(N);

    createLargeDataset(h_A, h_B, N);

    // test CPU implementation
    std::cout << "Running CPU...\n";
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vec_add_host(h_A.data(), h_B.data(), h_C_host.data(), N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
    std::cout << "End CPU\n";

    // test gpu implementation
    std::cout << "Running GPU...\n";
    auto gpu_start = std::chrono::high_resolution_clock::now();
    vec_add_device(h_A.data(), h_B.data(), h_C_device.data(), N);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count();
    std::cout << "End GPU\n";

    // Results
    std::cout << "\n========== Results ==========\n";
    std::cout << "Array size: " << N << " elements\n";
    std::cout << "CPU time: " << cpu_time << " ms\n";
    std::cout << "GPU time: " << gpu_time << " ms\n";
    std::cout << "Speedup: " << (float)cpu_time / gpu_time << "x\n";

    return 0;
}