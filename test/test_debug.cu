#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>

// Simple kernel for testing
__global__ void hgemm_test_kernel(half *a, half *b, half *c, int M, int N, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < M && x < N)
    {
        float res = 0.0f;
        for (int i = 0; i < K; i++)
        {
            res += __half2float(a[y * K + i]) * __half2float(b[i * N + x]);
        }
        c[y * N + x] = __float2half(res);
    }
}

void hgemm_test(half *a, half *b, half *c, int M, int N, int K)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    hgemm_test_kernel<<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaDeviceSynchronize();
}

int main() {
    // Small test case
    int M = 2, N = 3, K = 2;

    // Allocate host memory
    half *h_a = (half*)malloc(M * K * sizeof(half));
    half *h_b = (half*)malloc(K * N * sizeof(half));
    half *h_c = (half*)malloc(M * N * sizeof(half));
    half *h_c_ref = (half*)malloc(M * N * sizeof(half));

    // Simple test data
    // A = [[1, 2], [3, 4]]
    h_a[0] = __float2half(1.0f);
    h_a[1] = __float2half(2.0f);
    h_a[2] = __float2half(3.0f);
    h_a[3] = __float2half(4.0f);

    // B = [[5, 6, 7], [8, 9, 10]]
    h_b[0] = __float2half(5.0f);
    h_b[1] = __float2half(6.0f);
    h_b[2] = __float2half(7.0f);
    h_b[3] = __float2half(8.0f);
    h_b[4] = __float2half(9.0f);
    h_b[5] = __float2half(10.0f);

    // Expected result C = A * B (row-major):
    // C[0][0] = 1*5 + 2*8 = 21
    // C[0][1] = 1*6 + 2*9 = 24
    // C[0][2] = 1*7 + 2*10 = 27
    // C[1][0] = 3*5 + 4*8 = 47
    // C[1][1] = 3*6 + 4*9 = 54
    // C[1][2] = 3*7 + 4*10 = 61

    float expected_C[6] = {21, 24, 27, 47, 54, 61};

    // Allocate device memory
    half *d_a, *d_b, *d_c, *d_c_ref;
    cudaMalloc(&d_a, M * K * sizeof(half));
    cudaMalloc(&d_b, K * N * sizeof(half));
    cudaMalloc(&d_c, M * N * sizeof(half));
    cudaMalloc(&d_c_ref, M * N * sizeof(half));

    cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // Run our kernel
    hgemm_test(d_a, d_b, d_c, M, N, K);

    // Run cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    half alpha = 1.f;
    half beta = 0.f;

    cublasGemmEx(handle,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_b, CUDA_R_16F, N,
                 d_a, CUDA_R_16F, K,
                 &beta,
                 d_c_ref, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaMemcpy(h_c, d_c, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_ref, d_c_ref, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    printf("A (2x2 row-major):\n");
    for (int i = 0; i < M * K; i++) {
        printf("%.1f ", __half2float(h_a[i]));
    }
    printf("\n\n");

    printf("B (2x3 row-major):\n");
    for (int i = 0; i < K * N; i++) {
        printf("%.1f ", __half2float(h_b[i]));
    }
    printf("\n\n");

    printf("Expected C = A * B (2x3 row-major):\n");
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < N; x++) {
            printf("%.1f ", expected_C[y * N + x]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Our kernel C:\n");
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < N; x++) {
            printf("%.1f ", __half2float(h_c[y * N + x]));
        }
        printf("\n");
    }
    printf("\n");

    printf("cuBLAS C_ref:\n");
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < N; x++) {
            printf("%.1f ", __half2float(h_c_ref[y * N + x]));
        }
        printf("\n");
    }
    printf("\n");

    printf("Comparison (kernel vs expected):\n");
    float max_error = 0.0f;
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < N; x++) {
            float kernel_val = __half2float(h_c[y * N + x]);
            float expected_val = expected_C[y * N + x];
            float error = fabsf(kernel_val - expected_val);
            max_error = fmax(max_error, error);
            printf("C[%d][%d]: kernel=%.1f, expected=%.1f, error=%.1f\n",
                   y, x, kernel_val, expected_val, error);
        }
    }
    printf("\nMax error vs expected: %.6f\n", max_error);

    printf("\nComparison (kernel vs cublas) - direct:\n");
    max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float kernel_val = __half2float(h_c[i]);
        float cublas_val = __half2float(h_c_ref[i]);
        float error = fabsf(kernel_val - cublas_val);
        max_error = fmax(max_error, error);
        printf("[%d]: kernel=%.1f, cublas=%.1f, error=%.1f\n",
               i, kernel_val, cublas_val, error);
    }
    printf("\nMax error vs cublas (direct): %.6f\n", max_error);

    printf("\nComparison (kernel vs cublas) - transpose:\n");
    max_error = 0.0f;
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < N; x++) {
            float kernel_val = __half2float(h_c[y * N + x]);
            float cublas_val = __half2float(h_c_ref[x * M + y]);
            float error = fabsf(kernel_val - cublas_val);
            max_error = fmax(max_error, error);
            printf("C[%d][%d]: kernel=%.1f, cublas[C^T][%d][%d]=%.1f, error=%.1f\n",
                   y, x, kernel_val, x, y, cublas_val, error);
        }
    }
    printf("\nMax error vs cublas (transpose): %.6f\n", max_error);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_ref);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_c_ref);
    cublasDestroy(handle);

    return 0;
}
