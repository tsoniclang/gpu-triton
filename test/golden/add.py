import triton
import triton.language as tl


@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, n, a_dim0, b_dim0, out_dim0, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    _t1 = (offsets) < (n)
    mask_a = offsets < a_dim0
    _t2 = tl.load(a_ptr + offsets, mask=mask_a & _t1, other=0.0)
    mask_b = offsets < b_dim0
    _t3 = tl.load(b_ptr + offsets, mask=mask_b & _t1, other=0.0)
    _t4 = (_t2) + (_t3)
    mask_out = offsets < out_dim0
    tl.store(out_ptr + offsets, _t4, mask=mask_out & _t1)


def add(a, b, out, n):
    a_dim0 = a.shape[0]
    b_dim0 = b.shape[0]
    out_dim0 = out.shape[0]
    grid = (triton.cdiv(out_dim0, 1024),)
    _add_kernel[grid](a, b, out, n, a_dim0, b_dim0, out_dim0, BLOCK_SIZE=1024)
