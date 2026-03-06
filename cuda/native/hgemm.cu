#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// 1.FP16
__global__ void hgemm_native_fp16_kernel(half *a, half *b, half *c, int M, int N, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < M && x < N)
    {
        // Use float for accumulator to reduce precision loss
        float res = 0.0f;
        for (int i = 0; i < K; i++)
        {
            res += __half2float(a[y * K + i]) * __half2float(b[i * N + x]);
        }
        c[y * N + x] = __float2half(res);
    }
}
// Host wrapper function to launch the kernel
void hgemm_native_fp16(half *a, half *b, half *c, int M, int N, int K)
{
    // Block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    hgemm_native_fp16_kernel<<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaDeviceSynchronize();
}
