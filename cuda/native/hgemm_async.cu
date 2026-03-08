#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// PTX
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca (cache all, L1 + L2); 支持 4,8,16bytes
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

// cg (cache global, L2); 只支持16字节
/**
 * cg 不经过L1 减少L1污染
 * shared.global 从gmem到smem
 * l2::128B 128字节对齐
 * 16 拷贝16字节
 */
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel(half *a, half *b, half *c, int M, int N, int K)
{ // 注意 这里的BK是16了
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    __shared__ half s_a[2][BK][BM], s_b[2][BK][BN];
    half r_load_a[TM];                       // 8
    half r_load_b[TN];                       // 8
    half r_comp_a[TM];                       // 8
    half r_comp_b[TN];                       // 8
    half r_c[TM][TN] = {__float2half(0.0f)}; // 8x8

    int load_smem_a_m = tid / 2;
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;
    int load_smem_b_k = (tid / 16);
    int load_smem_b_n = (tid % 16) * 8;

    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    if (load_gmem_a_m >= M || load_gmem_b_n >= N)
        return;
    // bk=0 buffer 0
    {
        int load_gmem_a_k = load_smem_a_k; // 第0块
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        LDST128BITS(s_b[0][load_smem_b_k][load_smem_b_n]) = LDST128BITS(b[load_gmem_b_addr]);
        LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]); // 因为s_a转置了
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
    }
    __syncthreads(); // 这里要sync一次
    // NOTE BK从1开始
    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk)
    {
        int smem_sel = (bk - 1) & 1;                 // bk 1->0, bk 2->1, bk 3->0, ...
        int smem_sel_next = bk & 1;                  // bk 1->1, bk 2->0, bk 3->1, ...
        int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]);
        LDST128BITS(r_load_b[0]) = LDST128BITS(b[load_gmem_b_addr]);
#pragma unroll
        for (int tk = 0; tk < BK; tk++)
        {
            LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
            LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);
// 一个线程的64个元素读取完毕
// 向量外积
#pragma unroll
            for (int tm = 0; tm < TM; tm++)
            {
#pragma unroll
                for (int tn = 0; tn < TN; tn++)
                {
                    r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

        // 对比非double buffers版本，此处不需要__syncthreads()，总共节省了
        // ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，HFMA计算
        // 使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。
        // 从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于
        // 加载下一块BK需要的数据到共享内存。
        // NOTE 注意r_load_a 的写入时机和同步时机
#pragma unroll
        for (int i = 0; i < 8; ++i)
        { // reg -> shared, fast
            s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
        LDST128BITS(s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]) = (LDST128BITS(r_load_b[0]));

        __syncthreads();
    }

// 计算剩下最后一块BK
#pragma unroll
    for (int tk = 0; tk < BK; tk++)
    {
        LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[1][tk][ty * TM]);
        LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[1][tk][tx * TN]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++)
        {
#pragma unroll
            for (int tn = 0; tn < TN; tn++)
            {
                r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int store_gmem_c_m = by * BM + ty * TM + i;
        int store_gmem_c_n = bx * BN + tx * TN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(c[store_gmem_c_addr]) = LDST128BITS(r_c[i][0]);
    }
}

// NOTE async 版本
// 优点 1. 实现全局内存到共享内存的直通路径 ，不用经过reg
// 2. 计算与数据传输的流水线重叠
// Load Data -> Wait for Data -> Store to Smem -> Compute
// 同步方法的线程在 Load 时会阻塞（Stall），直到数据从 DRAM 返回，这段时间计算单元（Tensor Core/CUDA Core）是空闲的。
// 3. cg 更好的缓存管理，不会污染L1
template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel(
    half *a, half *b, half *c, int M, int N, int K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + tx; // tid within the block
    // 2*128*16*2=8KB, 2*16*128*2=8KB
    __shared__ half s_a[2][BK][BM + OFFSET];
    __shared__ half s_b[2][BK][BN + OFFSET];
    half r_load_a[TM];                           // 8
    half r_comp_a[TM];                           // 8
    half r_comp_b[TN];                           // 8
    half r_c[TM][TN] = {__float2half(0.0f)};     // 8x8
    int load_smem_a_m = tid / 2;                 // row 0~127
    int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
    int load_smem_b_k = tid / 16;                // row 0~15
    int load_smem_b_n = (tid % 16) * 8;          // col 0,8,...,120
    int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
    if (load_gmem_a_m >= M || load_gmem_b_n >= N)
        return;
    // bk0
    {
        int load_gmem_a_k = load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[0][load_smem_b_k][load_smem_b_n]);
        /** NOTE
         * 在CUDA C++中，&s_b[...] 得到的是一个通用的64位指针（Generic Pointer）。
         * 但是，底层的 PTX 指令 cp.async 需要的是 共享内存空间内的32位偏移量（offset），
         * 而不是通用的虚拟内存地址。
         */
        CP_ASYNC_CG(load_smem_b_ptr, &b[load_gmem_b_addr], 16);
        /**
         * 将之前发出的所有 cp.async 指令标记为一个“组”（Group）。
         * 硬件会跟踪这个组内的所有拷贝操作是否完成。如果不调用这个，
         * 后续的 WAIT_GROUP 就无法区分哪些指令是需要等待的。
         */
        CP_ASYNC_COMMIT_GROUP();
        // load 8 half in 1 memory issue.
        LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]);
        LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        { // reg -> shared, fast
            s_a[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
        /**
         *  表示等待直到剩余的未完成组的数量为 0。
         */
        CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();
    for (int bk = 1; bk < (K + BK - 1) / BK; ++bk)
    {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;
        int load_gmem_a_k = bk * BK + load_smem_a_k; // global col of a
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = bk * BK + load_smem_b_k; // global row of b
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

        uint32_t load_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[smem_sel_next][load_smem_b_k][load_smem_b_n]);
        CP_ASYNC_CG(load_smem_b_ptr, &b[load_gmem_b_addr], 16);
        CP_ASYNC_COMMIT_GROUP();
#pragma unroll
        for (int tk = 0; tk < BK; tk++)
        {
            LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[smem_sel][tk][ty * TM]);
            LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[smem_sel][tk][tx * TN]);
// 一个线程的64个元素读取完毕
// 向量外积
#pragma unroll
            for (int tm = 0; tm < TM; tm++)
            {
#pragma unroll
                for (int tn = 0; tn < TN; tn++)
                {
                    r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }
        LDST128BITS(r_load_a[0]) = LDST128BITS(a[load_gmem_a_addr]);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        { // reg -> shared, fast
            s_a[smem_sel_next][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }
#pragma unroll
    for (int tk = 0; tk < BK; tk++)
    {
        LDST128BITS(r_comp_a[0]) = LDST128BITS(s_a[1][tk][ty * TM]);
        LDST128BITS(r_comp_b[0]) = LDST128BITS(s_b[1][tk][tx * TN]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++)
        {
#pragma unroll
            for (int tn = 0; tn < TN; tn++)
            {
                r_c[tm][tn] = __hfma(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int store_gmem_c_m = by * BM + ty * TM + i;
        int store_gmem_c_n = bx * BN + tx * TN;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        LDST128BITS(c[store_gmem_c_addr]) = LDST128BITS(r_c[i][0]);
    }
}

// Host wrapper function for hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf(half *a, half *b, half *c, int M, int N, int K)
{
    // Thread block dimensions: 16x16 = 256 threads
    dim3 block(16, 16);
    // Grid dimensions
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);

    // Launch kernel with default template parameters (BM=128, BN=128, BK=16, TM=8, TN=8)
    hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel<128, 128, 16, 8, 8, 0>
        <<<grid, block>>>(a, b, c, M, N, K);
}

// Host wrapper function for hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel
void hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async(half *a, half *b, half *c, int M, int N, int K)
{
    // Thread block dimensions: 16x16 = 256 threads
    dim3 block(16, 16);
    // Grid dimensions
    dim3 grid((N + 128 - 1) / 128, (M + 128 - 1) / 128);

    // Launch kernel with default template parameters (BM=128, BN=128, BK=16, TM=8, TN=8)
    hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel<128, 128, 16, 8, 8, 0>
        <<<grid, block>>>(a, b, c, M, N, K);
}