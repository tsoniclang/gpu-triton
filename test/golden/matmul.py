import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = k_block * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        a_tile = tl.load(a_ptr + offs_m[:, None] * K + k_offsets[None, :], mask=a_mask, other=0.0)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
        b_tile = tl.load(b_ptr + k_offsets[:, None] * N + offs_n[None, :], mask=b_mask, other=0.0)
        acc = acc + tl.dot(a_tile, b_tile)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=c_mask)


def matmul(a, b, c):
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _matmul_kernel[grid](a, b, c, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32)
