#include <iostream>
#include <chrono>
#include <cstring>

#include "matrix.h"
#include "cpufunc.h"


using namespace std::chrono;
int main() {
    size_t M = SIZE;
    size_t K = 2 * SIZE;
    size_t N = 3 * SIZE;

    size_t flop_count = 2 * K * M * N;
    size_t byte_count = (M * K + N * K + M * N) * sizeof(TestType);


    std::cout<<"Matrix Properties:"<<std::endl;
    std::cout<<"A is ("<<M<<"x"<<K<<")"<<std::endl;
    std::cout<<"B is ("<<K<<"x"<<N<<")"<<std::endl;
    std::cout<<"C is ("<<M<<"x"<<N<<")"<<std::endl;
    std::cout<<"Required Arithmetic: "<<(double)flop_count/1000000000.0<<" GFLOP"<<std::endl;
    std::cout<<"Required Memory Access: "<<(double)byte_count/1000000000.0<<" GB"<<std::endl;
    std::cout<<"Arithmetic Intensity: "<<(double)flop_count / (double)byte_count<< " FLOP/B"<<std::endl;

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int bus_width = prop.memoryBusWidth;
    int clock_rate_khz;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrMemoryClockRate, 0);
    std::cout<<"Theoretical Memory Bandwidth: "<< (bus_width / 8) * clock_rate_khz / 1000000.0<<" GB/s"<<std::endl;
    std::cout << "Number of Streaming Multiprocessors: " << prop.multiProcessorCount << std::endl;

    std::cout<<std::endl;

    typedef high_resolution_clock clock;

    TestType* A = GenerateMatrix<TestType>(M, K);           // create random matrices to multiply
    TestType* B = GenerateMatrix<TestType>(K, N);

    TestType* cpu_C = new TestType[M * N];                     // allocate space for the output matrix

    if (CPU) {
        auto t_cpu_start = clock::now();
        cpuMultiply<TestType>(cpu_C, A, B, M, N, K);
        auto t_cpu_end = clock::now();

        // output CPU timer results
        milliseconds t_cpu = duration_cast<milliseconds>(t_cpu_end - t_cpu_start);
        std::cout<<"CPU single-thread:   "<<t_cpu.count()<<" ms (" <<
            (double)flop_count / (double)t_cpu.count() / 1000000.0 <<" GFLOP/S)"<<
                std::endl;
    }

    TestType* C = new TestType[M * N];
    TestType canonical_error = 0;
    TestType tile_error = 0;
    TestType cublas_error = 0;

    auto t_canonical_start = clock::now();
    if (sizeof(TestType) == sizeof(double)) {
        gpuMultiply_Canonical_double((double*)C, (double*)A, (double*)B, M, N, K);
    }
    else if (sizeof(TestType) == sizeof(float)) {
        gpuMultiply_Canonical_float((float*)C, (float*)A, (float*)B, M, N, K);
    }
    auto t_canonical_end = clock::now();
    if (CPU) canonical_error = Compare(cpu_C, C, M*N);
    memset(C, 0, M * N * sizeof(TestType));

    milliseconds t_canonical = duration_cast<milliseconds>(t_canonical_end - t_canonical_start);
    std::cout << "GPU canonical:       " << t_canonical.count() << " ms (" <<
        (double)flop_count / (double)t_canonical.count() / 1000000.0 << " GFLOP/S)";
    if (CPU) std::cout << "\t\t\t (E = " << canonical_error << ")";
    std::cout << std::endl;







    auto t_tile_start = clock::now();
    if (sizeof(TestType) == sizeof(double)) {
        gpuMultiply_Tile_double((double*)C, (double*)A, (double*)B, M, N, K);
    }
    else if (sizeof(TestType) == sizeof(float)) {
        gpuMultiply_Tile_float((float*)C, (float*)A, (float*)B, M, N, K);
    }
    auto t_tile_end = clock::now();
    if (CPU) tile_error = Compare(cpu_C, C, M*N);
    memset(C, 0, M * N * sizeof(TestType));

    milliseconds t_tile = duration_cast<milliseconds>(t_tile_end - t_tile_start);
    std::cout << "GPU tile:            " << t_tile.count() << " ms (" <<
        (double)flop_count / (double)t_tile.count() / 1000000.0 << " GFLOP/S)";
    if (CPU) std::cout << "\t\t\t (E = " << tile_error << ")";
    std::cout << std::endl;






    auto t_cublas_start = clock::now();
    gpuMultiply_cublas<TestType>(C, A, B, M, N, K);
    if (CPU) cublas_error = Compare(cpu_C, C, M*N);
    memset(C, 0, M * N * sizeof(TestType));
    auto t_cublas_end = clock::now();

    milliseconds t_cublas = duration_cast<milliseconds>(t_cublas_end - t_cublas_start);
    std::cout<<"GPU cuBLAS:          "<<t_cublas.count()<<" ms ("<<
        (double)flop_count / (double)t_cublas.count() / 1000000.0 <<" GFLOP/S)";
    if (CPU) std::cout<<"\t\t\t (E = "<<cublas_error<<")";
    std::cout<<std::endl;


    delete[] A;
    delete[] B;
    delete[] C;
    delete[] cpu_C;

    return 0;
}
