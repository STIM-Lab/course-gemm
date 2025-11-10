#pragma once
#include <random>


// Note: All matrices are represented in column-major order


/**
 * These variables can be modified for testing.
 * TILE_K defines the tile size along the K dimension (columns in A and rows in B)
 * TestType is the data type of the matrices to be multiplied (currently supports 32-bit and 64-bit floating point)
 * CPU is a flag that determines if the CPU version is run (it gets really slow for large matrices)
 * SIZE_APPROX is the approximate size of the smallest matrix dimension (the implementation requires that this is divisible by K)
 */
#define TILE_K 32
typedef float TestType;
static bool CPU = false;
#define SIZE_APPROX 400

/**
 * Everything below this point are declarations that shouldn't be changed unless you know what you're doing
 */

// ensure that the K dimension is divisible by TILE_K
#define SIZE (SIZE_APPROX / TILE_K) * TILE_K

// Matrix-Matrix multiplication compile-time arguments for testing
#define TILE_M 32
#define TILE_N 32


void gpuMultiply_Canonical_float(float* C, const float* A, const float* B, size_t M, size_t N, size_t K);
void gpuMultiply_Canonical_double(double* C, const double* A, const double* B, size_t M, size_t N, size_t K);
void gpuMultiply_Tile_float(float* C, const float* A, const float* B, size_t M, size_t N, size_t K);
void gpuMultiply_Tile_double(double* C, const double* A, const double* B, size_t M, size_t N, size_t K);
