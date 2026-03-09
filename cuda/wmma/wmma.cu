#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

using namespace nvcuda;
// NOTE 1.namespace
#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n", ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n", ::"r"(dst), "l"(src), "n"(bytes))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// NOTE only 1 warp per block(32 threads), m16n16k16. A, B, C: all row_major.
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
__global__ void hgemm_wmma_m16n16k16_native_kernel(half *A, half *B, half *C, int M, int N,
                                                   int K)
{
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    const int load_gmem_a_m = blockIdx.y * WMMA_M;
    const int load_gmem_b_n = blockIdx.x * WMMA_N;
    if (load_gmem_a_m >= M || load_gmem_b_n >= N)
        return;
    wmma::fragment<wmma::accumulator, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, 0.0);
#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k)
    {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_fra
            g;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_majot> B_fra
            g;
        // load
        // wmma::load_matrix_sync(frag, ptr, ldm);
        // ldm : leading dimension / strides 跨度
        wmma::load_matrix_sync(A_frag, A + load_gmem_a_m * K + k * WMMA_K, K);
        wmma::load_matrix_synx(B_frag, B + (K * WMMA_K) * N + load_gmem_b_n, N);
        wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        __syncthreads();
    }
    // NOTE 这里必须显式指定 layout (wmma::mem_row_major 或 mem_col_major)，因为 Accumulator 本身没有布局概念。
    // wmma::row_major 描述的是矩阵乘法的数学语义——矩阵 A 和 B 在乘法中的角色：
    // wmma::mem_row_major 描述的是内存中的实际存储方式——加载/存储时的数据排布：
    wmma::store_matrix_sync(C + load_gmem_a_m * N + load_gmem_b_n, C_frag, N, wmma::mem_row_major);
}

// m16n16k16 wmma  + tile MMA with smem,  A, B, C: all row_major.
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16,
          const int WMMA_TILE_M = 4, const int WMMA_TILE_N = 2>
__global__ void hgemm_wmma_m16n16k16_mma4x2_kernel(half *A, half *B, half *C, int M, int N, int K)
{
    // NOTE 一个block 有256个线程 有8个warp
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    constexpr int NUM_K_TILE = div_ceil(K, WMMA_K);
    constexpr int BM = WMMA_M * WMMA_TILE_M; // 16 * 4 = 64;
    constexpr int BN = WMMN_N * WMMN_TILE_N; // 16 * 2 = 32;
    constexpr int BK = WMMA_K;
    __shared__ half s_a[BM][BK], s_b[BK][BN];
    // NOTE  要保证相同的warp下thread执行相同的指令
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const warp_id = tid / WARP_SIZE;
    const lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id / WMMA_TILE_N;
    const int warp_n = warp_id % WMMA_TILE_N;
    // NOTE 因为线程和smem的一一对应关系，
    /**
     * 64*16/256=4, half4, 16x32/256=2, half2
     * s_a, 64*16, 每个线程load 4 half, 每行需要4线程，64行，共256线程
     */
    const int load_smem_a_m = tid / 4;
    const int load_smem_a_k = (tid % 4) * 4;
    const int load_smem_b_k = tid / 16;
    const int load_smem_b_n = (tid % 16) * 2;
    if (load_gmem_a_m >= M || load_gmem_b_n >= N)
        return;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment<C_frag, 0.0>;
#pragma unroll
    for (int k = 0; k < NUM_K_TILES; ++k)
    {
        int load_gmem_a_k = k * WMMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        LDST64BITS(s_a[load_smem_a_m][load_smem_a_k]) = (LDST64BITS(A[load_gmem_a_addr]));
        LDST32BITS(s_b[load_smem_b_k][load_smem_b_n]) = (LDST32BITS(B[load_gmem_b_addr]));
        __syncthreads();
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;
        // NOTE 这里可知，load_matrix_sync 可以从全局内存读，也可以从smem读
        wmma::load_matrix_sync(A_frag, &s_a[warp_m * WMMA_M][0], BK);
        wmma::load_matrix_sync(B_frag, &s_b[0][warp_n * WMMA_N], BN);
        wmma::mma_sync(C_frag, A_frag, B_frag, C_grag);
        __syncthreads();
    }
    const int store_gmem_a_m = by * BM + warp_m * WMMA_M;
    const int store_gmem_a_n = bx * BN + warp_n * WMMA_N;
    wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag, N, wmma::mem_row_major);
}
