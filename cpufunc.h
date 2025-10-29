#pragma once

#include <matrix.h>

#include <cublas_v2.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

#include <tira/cuda/error.h>

// generate a random NxN matrix in host memory
template<typename Type>
Type* GenerateMatrix(size_t N, size_t M) {
    Type* result = new Type[N*M];                         // allocate space to store the matrix

    std::mt19937 mt;                                            // create a Mersenne twister instance
    std::uniform_real_distribution<Type> dist(0, 1);      // create a real distribution between (0, 1)

    for (size_t i = 0; i < N*M; i++) {                            // for each element in the NxN array
        result[i] = dist(mt);                                  // create a random number from the specified distribution
    }
    return result;
}

// matrix multiplication using the CPU
template<typename Type>
void cpuMultiply(Type* C, const Type* A, const Type* B, size_t M, size_t N, size_t K) {
    for (size_t mi = 0; mi < M; mi++) {
        for (size_t ni = 0; ni < N; ni++) {
            TestType inner = 0.0;
            for (size_t ki = 0; ki < K; ki++) {
                inner += A[ki * M + mi] * B[ni * K + ki];
            }
            C[ni * M + mi] = inner;
        }
    }
}

// Compare matrices to calculate the maximum error
template<typename Type>
Type Compare(Type* C0, Type* C1, size_t S) {
    Type max_abs_error = 0.0;
    for (size_t i = 0; i < S; i++) {
        Type abs_error = std::abs(C0[i] - C1[i]);
        max_abs_error = std::max(max_abs_error, abs_error);
    }
    return max_abs_error;
}


/**
 *
 * @brief Calculate a Matrix-Matrix multiplication using cuBLAS. In most cases, this will be the most efficient
 *      implementation.
 */
template<typename Type>
void gpuMultiply_cublas(Type* C, const Type* A, const Type* B, size_t M, size_t N, size_t K) {

    // Allocate space on the GPU and copy all of the required matrices
    Type* gpu_C;
    cudaMalloc((void**)&gpu_C, M * N * sizeof(Type));

    Type* gpu_A;
    cudaMalloc((void**)&gpu_A, M * K * sizeof(Type));
    cudaMemcpy(gpu_A, A, M * K * sizeof(Type), cudaMemcpyHostToDevice);

    Type* gpu_B;
    cudaMalloc((void**)&gpu_B, K * N * sizeof(Type));
    cudaMemcpy(gpu_B, B, K * N * sizeof(Type), cudaMemcpyHostToDevice);

    // Create a cuBLAS handle and initialize
    cublasHandle_t handle;
    cublasCreate(&handle);

    // The alpha and beta parameters are required for gemm, but they aren't really used so I set them to 1.0
    Type alpha = 1.0f;
    Type beta = 0.0f;

    // Because the BLAS standard has different functions for different data types, I used preprocessor
    //  directives to call the correct function during compile time. This can be done with some fancy
    //  C++ template stuff, but I wanted to keep it simple.
    if (sizeof(Type) == sizeof(double)) {
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (double*)&alpha, (double*)gpu_A, M, (double*)gpu_B, K, (double*)&beta, (double*)gpu_C, M);
    }
    else if (sizeof(Type) == sizeof(float)) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, (float*)&alpha, (float*)gpu_A, M, (float*)gpu_B, K, (float*)&beta, (float*)gpu_C, M);
    }

    cudaMemcpy(C, gpu_C, M * N * sizeof(TestType), cudaMemcpyDeviceToHost);

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
}