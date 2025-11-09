#include <matrix.h>

//handle error macro
static void cuHandleError( const cudaError_t err, const char *file,  const int line ) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ),  file, line );

    }
}
#define HANDLE_ERROR( err ) (cuHandleError( err, __FILE__, __LINE__ ))

/**
 * @brief CUDA kernel that performs a naive matrix multiplication, assuming that one thread is launched
 *          for each scalar value in the output matrix
 * @param C is a pointer to the output matrix (result of AxB)
 * @param A is a pointer to the right-hand-side operand
 * @param B is a pointer to the left-hand-side operand
 * @param M column size of matrix A (and C)
 * @param N row size of matrix B (and C)
 * @param K row size of matrix A, column size of matrix B
 */
template<typename Type>
__global__ void kernelMultiply_Canonical(Type* C, const Type* A, const Type* B, size_t M, size_t N, size_t K) {
    // calculate the element of the C that is calculated by this thread
    size_t mi = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ni = blockIdx.y * blockDim.y + threadIdx.y;

    // make sure that the thread is evaluating a value inside C
    if (mi >= M || ni >= N) return;

    // create a register to store the accumulated value of the inner product used to evaluate C
    Type inner = 0.0;

    // for each element in the inner product (row of A, column of B)
    for (size_t ki = 0; ki < K; ki++) {

        // perform both fetches and perform the FMA
        inner += A[ki * M + mi] * B[ni * K + ki];
    }

    // write the final inner product value to the corresponding address in the output matrix
    C[ni * M + mi] = inner;
}

/**
 *
 * @brief Host function that performs all of the initialization and kernel calls necessary to perform
 *      a canonical matrix multiplication.
 */
template<typename Type>
void gpuMultiply_Canonical(Type* C, const Type* A, const Type* B, size_t M, size_t N, size_t K) {

    // Allocate space for the output matrix on the device
    Type* gpu_C;
    HANDLE_ERROR(cudaMalloc(&gpu_C, M * N * sizeof(Type)));

    // Allocate space for both operands on the device and copy the matrices from the host
    Type* gpu_A;
    HANDLE_ERROR(cudaMalloc(&gpu_A, M * K * sizeof(Type)));
    HANDLE_ERROR(cudaMemcpy(gpu_A, A, M * K * sizeof(Type), cudaMemcpyHostToDevice));

    Type* gpu_B;
    HANDLE_ERROR(cudaMalloc(&gpu_B, K * N * sizeof(Type)));
    HANDLE_ERROR(cudaMemcpy(gpu_B, B, K * N * sizeof(Type), cudaMemcpyHostToDevice));

    // Get the device properties required to optimize the grid configuration. This code
    //  calculates the maximum number of threads supported per block. The configuration uses
    //  a 2D block dimension, where each dimension is the square root of the maximum block
    //  size. Note: this is almost always 32 (max threads/block of 1024) on modern GPUs.
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    int threads_per_block = prop.maxThreadsPerBlock;
    int sqrt_threads_per_block = std::sqrt(threads_per_block);
    dim3 block_dim(sqrt_threads_per_block, sqrt_threads_per_block);
    dim3 grid_dim(M / block_dim.x + 1, N / block_dim.y + 1);

    // Launch the kernel with the specified grid parameters
    kernelMultiply_Canonical<Type><<<grid_dim, block_dim>>>(gpu_C, gpu_A, gpu_B, M, N, K);

    // Copy the resulting matrix C from the GPU to host memory
    cudaMemcpy(C, gpu_C, M * N * sizeof(Type), cudaMemcpyDeviceToHost);

    // Free all of the GPU memory
    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
}

/**
 *
 * @brief CUDA kernel function for performing a tiled matrix multiplication. Here the tile size is specified by
 *      the block dimensions along M and N. The tile size for the K dimension is set using a defined constant
 *      in the matrix.h header file.
 */
template<typename Type>
__global__ void kernelMultiply_Tile(Type* C, const Type* A, const Type* B, size_t M, size_t N, size_t K) {
    size_t mi = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ni = blockIdx.y * blockDim.y + threadIdx.y;
    if (mi >= M || ni >= N) return;

    Type inner = 0.0f;

    // The main difference is that this code uses an additional outer loop to iterate across tiles.
    for (size_t ki_ = 0; ki_ < K; ki_+= TILE_K) {

        // This inner loop is responsible for evaluating the part of the inner product within the tile.
        //  This loop will be unrolled by the compiler, however different unrolling parameters can be specified
        //  by using a #pragma unroll n preprocessor directive.
        for (size_t ki = 0; ki < TILE_K; ki++) {
            size_t k = ki_ + ki;

            inner += A[k * M + mi] * B[ni * K + k];
        }
    }

    C[ni * M + mi] = inner;
}
template<typename Type>
void gpuMultiply_Tile(Type* C, const Type* A, const Type* B, size_t M, size_t N, size_t K) {

    Type* gpu_C;
    cudaMalloc(&gpu_C, M * N * sizeof(Type));

    Type* gpu_A;
    cudaMalloc(&gpu_A, M * K * sizeof(Type));
    cudaMemcpy(gpu_A, A, M * K * sizeof(Type), cudaMemcpyHostToDevice);

    Type* gpu_B;
    cudaMalloc(&gpu_B, K * N * sizeof(Type));
    cudaMemcpy(gpu_B, B, K * N * sizeof(Type), cudaMemcpyHostToDevice);

    // The main difference in this code is that constants for the tile sizes are used to set the grid dimensions.
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    int threads_per_block = prop.maxThreadsPerBlock;
    int sqrt_threads_per_block = std::sqrt(threads_per_block);
    dim3 block_dim(sqrt_threads_per_block, sqrt_threads_per_block);
    
    dim3 grid_dim(M / block_dim.x + 1, N / block_dim.y + 1);

    // Launch the tiled kernel with the specified grid parameters
    kernelMultiply_Tile<Type><<<grid_dim, block_dim>>>(gpu_C, gpu_A, gpu_B, M, N, K);

    cudaMemcpy(C, gpu_C, M * N * sizeof(Type), cudaMemcpyDeviceToHost);

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
}


void gpuMultiply_Canonical_float(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
	gpuMultiply_Canonical<float>(C, A, B, M, N, K);
}

void gpuMultiply_Canonical_double(double* C, const double* A, const double* B, size_t M, size_t N, size_t K) {
	gpuMultiply_Canonical<double>(C, A, B, M, N, K);
}

void gpuMultiply_Tile_float(float* C, const float* A, const float* B, size_t M, size_t N, size_t K) {
	gpuMultiply_Tile<float>(C, A, B, M, N, K);
}

void gpuMultiply_Tile_double(double* C, const double* A, const double* B, size_t M, size_t N, size_t K) {
	gpuMultiply_Tile<double>(C, A, B, M, N, K);
}
