#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward declarations for host wrapper functions
extern void hgemm_native_fp16(half *a, half *b, half *c, int M, int N, int K);
extern void hgemm_sliced_k_fp16(half *a, half *b, half *c, int M, int N, int K);

#include "../cuda/native/hgemm.cu"
#include "../utils/test_utils.h"

// Test configurations
const int MNK_CONFIGS[][3] = {
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096},
    {8192, 8192, 8192},

};

const int NUM_CONFIGS = sizeof(MNK_CONFIGS) / sizeof(MNK_CONFIGS[0]);
const int REPEAT = 10;
const int WARMUP = 2;

void print_separator()
{
    printf("================================================================================\n");
}

// Update function pointer type to match kernel signature
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
        // Run error check on a small configuration
        int M = 512, N = 512, K = 512;
        float max_error = gemm_error_check_nn<half>(kernel, M, N, K);
        printf("Error check (M=N=K=%d): max_error = %.6f\n", M, max_error);
        // FP16 has limited precision, use a more lenient threshold
        if (max_error < 0.1f)
        {
            printf("✓ PASSED: Error within acceptable threshold (FP16 precision)\n");
        }
        else
        {
            printf("✗ FAILED: Error exceeds threshold\n");
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

    // Test native FP16 kernel
    test_kernel("hgemm_native_fp16", hgemm_native_fp16);

    // Test sliced K FP16 kernel
    test_kernel("hgemm_sliced_k_fp16", hgemm_sliced_k_fp16);

    printf("All tests completed!\n");
    print_separator();

    return 0;
}
