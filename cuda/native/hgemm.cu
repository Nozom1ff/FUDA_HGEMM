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

// 共享内存 + 分块版本
template <const int BM = 16, const int BN = 16, const int BK = 16>
__global__ void hgemm_sliced_k_fp16_kernel(half *a, half *b, half *c, int M, int N, int K)
{
    // K方向一个tile 一个tile地读取 相当于一个线程处理一个数据 线程数 = BM * BN
    __shared__ half ma[BM][BK], mb[BK][BN];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    // 计算全局输出坐标
    int row = by * BM + ty;
    int col = bx * BN + tx;

    // 边界检查：只处理有效输出
    if (row >= M || col >= N)
        return;

    // 使用float累积以减少精度损失
    float sum = 0.0f;

    for (int k = 0; k < (K + BK - 1) / BK; ++k)
    {
        // 加载A分块到共享内存
        // ma是 BM x BK，blockDim是 BM x BN = BM x BM (因为 BN = BM)
        // 每个线程需要加载 BK/BM 个元素（如果 BK > BM）或每个线程加载1个
        // 这里简化：让每个线程按顺序加载
        int load_size = BM * BK;
        for (int i = tid; i < load_size; i += blockDim.x * blockDim.y)
        {
            int ma_row = i / BK;
            int ma_col = i % BK;
            int g_row = by * BM + ma_row;
            int g_col = k * BK + ma_col;
            if (g_row < M && g_col < K)
                ma[ma_row][ma_col] = a[g_row * K + g_col];
            else
                ma[ma_row][ma_col] = __float2half(0.0f);
        }

        // 加载B分块到共享内存
        // mb是 BK x BN
        int load_size_b = BK * BN;
        for (int i = tid; i < load_size_b; i += blockDim.x * blockDim.y)
        {
            int mb_row = i / BN;
            int mb_col = i % BN;
            int g_row = k * BK + mb_row;
            int g_col = bx * BN + mb_col;
            if (g_row < K && g_col < N)
                mb[mb_row][mb_col] = b[g_row * N + g_col];
            else
                mb[mb_row][mb_col] = __float2half(0.0f);
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BK; ++i)
        {
            sum += __half2float(ma[ty][i]) * __half2float(mb[i][tx]);
        }
        __syncthreads();
    }
    c[row * N + col] = __float2half(sum);
}

// 启动函数
// Host wrapper function to launch the sliced_k kernel
void hgemm_sliced_k_fp16(half *a, half *b, half *c, int M, int N, int K)
{
    const int BM = 16;
    const int BN = 16;
    const int BK = 16;

    dim3 blockDim(BN, BM);  // 16x16 = 256 threads
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_sliced_k_fp16_kernel<BM, BN, BK><<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaDeviceSynchronize();
}

// Host wrapper function to launch the native kernel
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
