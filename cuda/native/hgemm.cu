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
template <const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void hgemm_sliced_k_fp16_kernel(half *a, half *b, half *c, int M, int N, int K)
{
    // K方向一个tile 一个tile地读取 相当于一个线程处理一个数据 线程数 = BM * BN
    __shared__ half ma[BM][BK], mb[BK][BN];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    // 线程重排：每个线程需要加载一个A元素和一个B元素
    // A分块是 BM x BK，需要 BM*BK 次加载
    // B分块是 BK x BN，需要 BK*BN 次加载
    // 总共 BM*BK + BK*BN 次加载
    int load_a_tid = tid;
    int load_b_tid = tid;

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
        // 加载A分块：ma[ty][tx]
        int ka = k * BK + tx;
        if (row < M && ka < K)
            ma[ty][tx] = a[row * K + ka];
        else
            ma[ty][tx] = __float2half(0.0f);

        // 加载B分块：mb[ty][tx]
        int kb = k * BK + ty;
        if (kb < K && col < N)
            mb[ty][tx] = b[kb * N + col];
        else
            mb[ty][tx] = __float2half(0.0f);

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

// Tiling + 8x8 Thread Tile + FP16向量化版本
// 每个线程计算 8x8 = 64 个输出元素
template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_kernel(half *a, half *b, half *c, int M, int N, int K)
{
    const int blockSize = 256;
    const int A_BLOCK_X = 8;  // 每个线程加载A的8个连续元素
    const int B_BLOCK_X = 32; // 每个线程加载B的32个连续元素
    const int C_BLOCK_X = 16; // 每个线程计算C的16个元素(4x4排列)

    __shared__ half sh_a[BM][BK];
    __shared__ half sh_b[BK][BN];

    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    int tid = threadIdx.x;

    int r0 = blockIdx_y * BM;
    int c0 = blockIdx_x * BN;

    // 计算线程在加载/计算时的坐标
    const int A_BLOCK_Y = blockSize / A_BLOCK_X; // 256 / 8 = 32
    const int B_BLOCK_Y = blockSize / B_BLOCK_X; // 256 / 32 = 8
    const int C_BLOCK_Y = blockSize / C_BLOCK_X; // 256 / 16 = 16

    int ay = tid / A_BLOCK_X;     // 0-31
    int ax = tid % A_BLOCK_X;     // 0-7
    int by_tid = tid / B_BLOCK_X; // 0-7
    int bx_tid = tid % B_BLOCK_X; // 0-31
    int cy = tid / C_BLOCK_X;     // 0-15
    int cx = tid % C_BLOCK_X;     // 0-15

    // 每个线程计算的输出块大小
    constexpr int Tm = BM / C_BLOCK_Y; // 128 / 16 = 8
    constexpr int Tn = BN / C_BLOCK_X; // 128 / 16 = 8

    // 累积器：每个线程计算 8x8 个输出
    float accum[TM][TN] = {0.0f};

    // K方向分块循环
    // 这个是泛化版本 可以特化 直接少走循环
    for (int k = 0; k < K; k += BK)
    {
// 加载A分块: sh_a[BM][BK]
#pragma unroll
        for (int i = ay; i < BM; i += A_BLOCK_Y)
        {
            int r = r0 + i;
#pragma unroll
            for (int j = ax; j < BK; j += A_BLOCK_X)
            {
                int c = k + j;
                sh_a[i][j] = (r < M && c < K) ? a[r * K + c] : __float2half(0.0f);
            }
        }

// 加载B分块: sh_b[BK][BN]
#pragma unroll
        for (int i = by_tid; i < BK; i += B_BLOCK_Y)
        {
            int r = k + i;
#pragma unroll
            for (int j = bx_tid; j < BN; j += B_BLOCK_X)
            {
                int c = c0 + j;
                sh_b[i][j] = (r < K && c < N) ? b[r * N + c] : __float2half(0.0f);
            }
        }

        __syncthreads();

// 计算：向量外积
#pragma unroll
        for (int p = 0; p < BK; ++p)
        {
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                int r = cy + i * C_BLOCK_Y;
                float a_val = __half2float(sh_a[r][p]);
#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    int c = cx + j * C_BLOCK_X;
                    accum[i][j] += a_val * __half2float(sh_b[p][c]);
                }
            }
        }

        __syncthreads();
    }

// 写回结果到全局内存
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int r = r0 + cy + i * C_BLOCK_Y;
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            int col_idx = c0 + cx + j * C_BLOCK_X;
            if (r < M && col_idx < N)
            {
                c[r * N + col_idx] = __float2half(accum[i][j]);
            }
        }
    }
}

// 特化的向量化访存版本
template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_kernel(half *a, half *b, half *c, int M, int N, int K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;  // tid within the block, 0-255
    __shared__ half s_a[BM][BK], s_b[BK][BN]; // 2*128*8*2=4KB

    // 每行八个数据 两个线程处理 一个线程读4个
    int sam = tid / 2;                // 0-127, which row in A to load
    int sak = (tid % 2 == 0) ? 0 : 4; // 向量化访存, 0 or 4
    int sbk = tid / 32;               // 0-7, which row in B to load
    int sbn = (tid % 32) * 4;         // 0, 4, 8, ..., 124, which col in B to load

    int gam = by * BM + sam;
    int gbn = bx * BN + sbn;

    // 计算每个线程负责的输出坐标
    int row = by * BM + ty * TM;
    int col = bx * BN + tx * TN;

    if (gam >= M || gbn >= N)
        return;

    float reg[TM][TN] = {0.0f};

    // K-loop
    for (int k = 0; k < (K + BK - 1) / BK; ++k)
    {
        // 加载A分块: s_a[BM][BK]
        int gak = k * BK + sak;
        if (gam < M && gak < K && sak < BK)
        {
            int a_addr = gam * K + gak;
            if (gak + 3 < K)
            {
                HALF2(s_a[sam][sak + 0]) = HALF2(a[a_addr + 0]);
                HALF2(s_a[sam][sak + 2]) = HALF2(a[a_addr + 2]);
            }
            else
            {
                for (int i = 0; i < 4 && gak + i < K; ++i)
                    s_a[sam][sak + i] = a[a_addr + i];
            }
        }
        else
        {
            for (int i = 0; i < 4; ++i)
                s_a[sam][sak + i] = __float2half(0.0f);
        }

        // 加载B分块: s_b[BK][BN]
        int gbk = k * BK + sbk;
        if (gbk < K && sbk < BK)
        {
            int b_addr = gbk * N + gbn;
            if (gbn + 3 < N)
            {
                HALF2(s_b[sbk][sbn + 0]) = HALF2(b[b_addr + 0]);
                HALF2(s_b[sbk][sbn + 2]) = HALF2(b[b_addr + 2]);
            }
            else
            {
                for (int i = 0; i < 4 && gbn + i < N; ++i)
                    s_b[sbk][sbn + i] = b[b_addr + i];
            }
        }
        else
        {
            for (int i = 0; i < 4; ++i)
                s_b[sbk][sbn + i] = __float2half(0.0f);
        }

        __syncthreads();

        // 计算
#pragma unroll
        for (int p = 0; p < BK; ++p)
        {
#pragma unroll
            for (int m = 0; m < TM; ++m)
            {
                int comp_smem_a_m = ty * TM + m;
                float a_val = __half2float(s_a[comp_smem_a_m][p]);
#pragma unroll
                for (int n = 0; n < TN; ++n)
                {
                    int comp_smem_b_n = tx * TN + n;
                    reg[m][n] += a_val * __half2float(s_b[p][comp_smem_b_n]);
                }
            }
        }

        __syncthreads();
    }

    // 写回结果
#pragma unroll
    for (int m = 0; m < TM; ++m)
    {
        int r = row + m;
        if (r < M)
        {
#pragma unroll
            for (int n = 0; n < TN; n += 4)
            {
                int col_idx = col + n;
                if (col_idx + 3 < N)
                    LDST64BITS(c[r * N + col_idx]) = LDST64BITS(reg[m][n]);
            }
        }
    }
}

// 真正优化版本：不同的索引计算方式（交错排列减少bank conflict）
template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_optimized_kernel(half *a, half *b, half *c, int M, int N, int K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;  // tid within the block, 0-255
    __shared__ half s_a[BM][BK], s_b[BK][BN]; // 2*128*8*2=4KB

    // 每行八个数据 两个线程处理 一个线程读4个
    int sam = tid / 2;                // 0-127, which row in A to load
    int sak = (tid % 2 == 0) ? 0 : 4; // 向量化访存, 0 or 4
    int sbk = tid / 32;               // 0-7, which row in B to load
    int sbn = (tid % 32) * 4;         // 0, 4, 8, ..., 124, which col in B to load

    int gam = by * BM + sam;
    int gbn = bx * BN + sbn;

    // 计算每个线程负责的输出坐标
    int row = by * BM + ty * TM;
    int col = bx * BN + tx * TN;

    if (gam >= M || gbn >= N)
        return;

    float reg[TM][TN] = {0.0f};

    // K-loop
    for (int k = 0; k < (K + BK - 1) / BK; ++k)
    {
        // 加载A分块: s_a[BM][BK]
        int gak = k * BK + sak;
        if (gam < M && gak < K && sak < BK)
        {
            int a_addr = gam * K + gak;
            if (gak + 3 < K)
            {
                HALF2(s_a[sam][sak + 0]) = HALF2(a[a_addr + 0]);
                HALF2(s_a[sam][sak + 2]) = HALF2(a[a_addr + 2]);
            }
            else
            {
                for (int i = 0; i < 4 && gak + i < K; ++i)
                    s_a[sam][sak + i] = a[a_addr + i];
            }
        }
        else
        {
            for (int i = 0; i < 4; ++i)
                s_a[sam][sak + i] = __float2half(0.0f);
        }

        // 加载B分块: s_b[BK][BN]
        int gbk = k * BK + sbk;
        if (gbk < K && sbk < BK)
        {
            int b_addr = gbk * N + gbn;
            if (gbn + 3 < N)
            {
                HALF2(s_b[sbk][sbn + 0]) = HALF2(b[b_addr + 0]);
                HALF2(s_b[sbk][sbn + 2]) = HALF2(b[b_addr + 2]);
            }
            else
            {
                for (int i = 0; i < 4 && gbn + i < N; ++i)
                    s_b[sbk][sbn + i] = b[b_addr + i];
            }
        }
        else
        {
            for (int i = 0; i < 4; ++i)
                s_b[sbk][sbn + i] = __float2half(0.0f);
        }

        __syncthreads();

        // 计算
#pragma unroll
        for (int p = 0; p < BK; ++p)
        {
#pragma unroll
            for (int m = 0; m < TM; ++m)
            {
                // 关键区别：不同的索引计算方式
                // 原版本: ty * TM + m = ty * 8 + m (连续排列)
                // 优化版: ty + m * 16 (交错排列，减少bank conflict)
                /**
                 *   "原版本没有严重bank conflict"的原因：

  1. ✅ Half类型的自然对齐：每2个half占4字节，正好1个bank
  2. ✅ 连续访问分散到不同bank：s_a[0], s_a[1], s_a[2], s_a[3] → bank 0,1,2,3
  3. ✅ 轻度2-way冲突可接受：现代GPU硬件可以很好地处理
  4. ✅ 空间局部性更重要：连续访问带来的性能提升远超过bank冲突的影响

  真正的性能瓶颈：
  - ❌ 不是bank conflict（只有2-way，很轻微）
  - ✅ 是全局内存带宽和访问合并
  - ✅ 是计算密度和指令级并行
  - ✅ 是共享内存的重用效率
                 */
                int comp_smem_a_m = ty + m * 16;
                float a_val = __half2float(s_a[comp_smem_a_m][p]);
#pragma unroll
                for (int n = 0; n < TN; ++n)
                {
                    int comp_smem_b_n = tx * TN + n;
                    reg[m][n] += a_val * __half2float(s_b[p][comp_smem_b_n]);
                }
            }
        }

        __syncthreads();
    }

    // 写回结果
#pragma unroll
    for (int m = 0; m < TM; ++m)
    {
        // 关键区别：对应不同的索引计算
        // 原版本: r = row + m = by*128 + ty*8 + m
        // 优化版: r = by*128 + (ty + m*16) = by*128 + ty + m*16
        int r = by * BM + ty + m * 16; // 对应 ty + m * 16
        if (r < M)
        {
#pragma unroll
            for (int n = 0; n < TN; ++n)
            {
                int col_idx = col + n;
                if (col_idx < N)
                    c[r * N + col_idx] = __float2half(reg[m][n]);
            }
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 8, const int TM = 8, const int TN = 8>
__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_kernel(half *a, half *b, half *c, int M, int N, int K)
{
    /**
     * 这个版本主要用float4 load/store 同时引入了fma
     * 完全仿照参考实现，使用float累加器减少精度损失
     */
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx;  // tid within the block, 0-255
    __shared__ half s_a[BM][BK], s_b[BK][BN]; // 2*128*8*2=4KB

    // 每行八个数据 两个线程处理 一个线程读4个
    int sam = tid / 2;                // 0-127, which row in A to load
    int sak = (tid % 2 == 0) ? 0 : 4; // 向量化访存, 0 or 4
    int sbk = tid / 32;               // 0-7, which row in B to load
    int sbn = (tid % 32) * 4;         // 0, 4, 8, ..., 124, which col in B to load

    int gam = by * BM + sam;
    int gbn = bx * BN + sbn;

    if (gam >= M || gbn >= N)
        return;

    // 使用float累加器来减少精度损失
    float r_c[TM][TN] = {0.0f}; // 8x8

    // K-loop
    for (int k = 0; k < (K + BK - 1) / BK; ++k)
    {
        // 加载A分块: s_a[BM][BK]
        int gak = k * BK + sak;
        int a_addr = gam * K + gak;
        LDST64BITS(s_a[sam][sak]) = LDST64BITS(a[a_addr]);

        // 加载B分块: s_b[BK][BN]
        int gbk = k * BK + sbk;
        int b_addr = gbk * N + gbn;
        LDST64BITS(s_b[sbk][sbn]) = LDST64BITS(b[b_addr]);

        __syncthreads();

        // 计算
#pragma unroll
        for (int k_tile = 0; k_tile < BK; k_tile++)
        {
#pragma unroll
            for (int m = 0; m < TM; m++)
            {
                int comp_smem_a_m = ty * TM + m;
                float a_val = __half2float(s_a[comp_smem_a_m][k_tile]);
#pragma unroll
                for (int n = 0; n < TN; n++)
                {
                    int comp_smem_b_n = tx * TN + n;
                    r_c[m][n] += a_val * __half2float(s_b[k_tile][comp_smem_b_n]);
                }
            }
        }
        __syncthreads();
    }

    // 写回结果
#pragma unroll
    for (int m = 0; m < TM; ++m)
    {
        int r = by * BM + ty * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n += 4)
        {
            int c_addr = r * N + (bx * BN + tx * TN + n);
            // 将float结果转换为half并存储
            half temp[4];
            temp[0] = __float2half(r_c[m][n]);
            temp[1] = __float2half(r_c[m][n + 1]);
            temp[2] = __float2half(r_c[m][n + 2]);
            temp[3] = __float2half(r_c[m][n + 3]);
            LDST64BITS(c[c_addr]) = LDST64BITS(temp[0]);
        }
    }
}

template <const int BM = 128,
          const int BN = 128,
          const int BK = 8,
          const int TM = 8,
          const int TN = 8,
          const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel(half *a, half *b, half *c, const int M, const int N, const int K)
{
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ half s_a[BK][BM + OFFSET]; // ！矩阵转置 从八路bank conflict 变为二路
    __shared__ half s_b[BK][BN + OFFSET];

    half r_load_a[TM / 2]; // 4 //为什么是TM/2? 不知道 但是手动一次读四个half 因为sa转置了不能想量化存储了
    half r_load_b[TN / 2]; // 4
    half r_comp_a[TM];
    half r_comp_b[TN];
    half r_c[TM][TN] = {__float2half(0.0f)};

    int sam = tid / 2;            // 0,1,2,...,127
    int sak = (tid & 2 - 1) << 2; // 0, 4
    int sbk = tid / 32;
    int sbn = (tid & 32 - 1) << 2; // 0, 4, 8, ..., 124
    int gam = by * BM + sam;
    int gbn = bx * BN + sbn;
    if (gam >= M || gbn >= N)
        return;
    for (int bk = 0; bk < (K + BK - 1) / BK; bk++)
    {
        int gak = bk * BK + sak;
        int a_addr = gam * K + gak;
        int gbk = bk * BK + sbk;
        int b_addr = gbk * N + bgn;
        LDST64BITS(r_load_a[0]) = LDST64BITS(a[a_addr]);
        LDST64BITS(r_load_b[0]) = LDST64BITS(b[b_arrr]);
        // 转置后放置
        s_a[sak][sam] = r_load_a[0];
        s_a[sak + 1][sam] = r_load_a[1];
        s_a[sak + 2][sam] = r_load_a[2];
        s_a[sak + 3][sam] = r_load_a[3];
        // b不需要
        LDST64BITS(s_b[sbk][sbn]) = LDST64BITS(r_load_b[0]);
        __syncthreads();
#pragma unroll
        for (int tk = 0; tk < BK; tk++)
        {
            LDST64BITS(r_comp_a[0]) = LDST64BITS(s_a[tk][ty * 4]);
            LDST64BITS(r_comp_a[4]) = LDST64BITS(s_a[tk][ty * 4 + 16 * 4]); // 向量化访存 M方向 16个线程16个线程的走
            LDST64BITS(r_comp_b[0]) = LDST64BITS(s_b[tk][tx * 4]);
            LDST64BITS(r_comp_b[4]) = LDST64BITS(s_b[tk][tx * 4 + 16 * 4]);
#pragma unroll
            for (int tm = 0; tm < TM; tm++)
            {
#pragma unroll
                for (int tn = 0; tn < TN; tn++)
                {
                    r_c = [tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
        __syncthreads();
    }
    // 写回 省略
}
// 启动函数

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

// Host wrapper function to launch the sliced_k kernel
void hgemm_sliced_k_fp16(half *a, half *b, half *c, int M, int N, int K)
{
    const int BM = 16;
    const int BN = 16;
    const int BK = 16; // 要保证线程和smem一一对应

    dim3 blockDim(BN, BM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_sliced_k_fp16_kernel<BM, BN, BK><<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaDeviceSynchronize();
}

// Host wrapper function to launch the t_8x8_sliced_k kernel
void hgemm_t_8x8_sliced_k(half *a, half *b, half *c, int M, int N, int K)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    // 1D线程块，256个线程
    dim3 blockDim(256);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaDeviceSynchronize();
}

// Host wrapper function to launch the t_8x8_sliced_k_f16x4 kernel
void hgemm_t_8x8_sliced_k_f16x4(half *a, half *b, half *c, int M, int N, int K)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    // 16x16 = 256 threads (2D)
    dim3 blockDim(16, 16);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k_f16x4_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaDeviceSynchronize();
}

// Host wrapper function to launch the t_8x8_sliced_k_f16x4_optimized kernel
void hgemm_t_8x8_sliced_k_f16x4_optimized(half *a, half *b, half *c, int M, int N, int K)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    // 16x16 = 256 threads (2D)
    dim3 blockDim(16, 16);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k_f16x4_optimized_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaDeviceSynchronize();
}

// Host wrapper function to launch the t_8x8_sliced_k_f16x4_pack kernel
void hgemm_t_8x8_sliced_k_f16x4_pack(half *a, half *b, half *c, int M, int N, int K)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    // 16x16 = 256 threads (2D)
    dim3 blockDim(16, 16);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    hgemm_t_8x8_sliced_k_f16x4_pack_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(a, b, c, M, N, K);
    cudaDeviceSynchronize();
}
