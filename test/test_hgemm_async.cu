#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward declarations for host wrapper functions
extern void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(half *a, half *b, half *c, int M, int N, int K);
extern void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(half *a, half *b, half *c, int M, int N, int K);

#include "../cuda/native/hgemm_async.cu"
#include "../utils/test_utils.h"

// Test configurations - dimensions must be divisible by BM=128, BN=128, BK=16
const int MNK_CONFIGS[][3] = {
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096},
    {16384, 16384, 16384},
};

const int NUM_CONFIGS = sizeof(MNK_CONFIGS) / sizeof(MNK_CONFIGS[0]);
const int REPEAT = 10;
const int WARMUP = 2;

void print_separator()
{
    printf("================================================================================\n");
}

// Function pointer type to match kernel signature
using hgemm_kernel_t = void (*)(half *, half *, half *, int, int, int);

void test_kernel(
    const char *kernel_name,
    hgemm_kernel_t kernel,
    bool check_error = true)
{
    print_separator();
    printf("Testing: %s\n", kernel_name);
    print_separator();

    if (check_error)
    {
        // Run error check on a small configuration (must be divisible by 128, 128, 16)
        int M = 512, N = 512, K = 64;
        float max_error = gemm_error_check_nn<half>(kernel, M, N, K);
        printf("Error check (M=%d, N=%d, K=%d): max_error = %.6f\n", M, N, K, max_error);
        // FP16 has limited precision, use appropriate threshold
        // The threshold is set to 0.5 because:
        // 1. FP16 has ~3-4 decimal digits of precision
        // 2. Accumulation of multiple FMA operations can accumulate rounding errors
        // 3. The difference in order of operations between our kernel and cuBLAS can cause small variations
        if (max_error < 0.5f)
        {
            printf("PASSED: Error within acceptable threshold (FP16 precision)\n");
        }
        else
        {
            printf("FAILED: Error exceeds threshold\n");
            return;
        }
        printf("\n");
    }

    // Performance benchmarks
    printf("%-12s %-12s %-12s %-12s %-12s\n", "M", "N", "K", "Time (sec)", "TFLOPS");
    printf("----------------------------------------------------------------------------\n");

    for (int i = 0; i < NUM_CONFIGS; i++)
    {
        int M = MNK_CONFIGS[i][0];
        int N = MNK_CONFIGS[i][1];
        int K = MNK_CONFIGS[i][2];

        float time_sec = perf_gemm<half>(kernel, M, N, K, REPEAT, WARMUP);
        float tflops = calculate_tflops(M, N, K, time_sec);

        printf("%-12d %-12d %-12d %-12.6f %-12.2f\n", M, N, K, time_sec, tflops);
    }
    printf("\n");
}

int main()
{
    // Set device
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    // Test double buffer kernel
    test_kernel("hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf", hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf, true);

    // Test async kernel
    test_kernel("hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async", hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async, true);

    printf("All tests completed!\n");
    print_separator();

    return 0;
}
