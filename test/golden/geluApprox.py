import triton
import triton.language as tl


@triton.jit
def _geluApprox_kernel(x_ptr, out_ptr, x_dim0, out_dim0, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_x = offsets < x_dim0
    v = tl.load(x_ptr + offsets, mask=mask_x, other=0.0)
    _t1 = 0.5
    _t2 = (_t1) * (v)
    _t3 = 1.0
    _t4 = 0.79788456
    _t5 = 0.044715
    _t6 = (_t5) * (v)
    _t7 = (_t6) * (v)
    _t8 = (_t7) * (v)
    _t9 = (v) + (_t8)
    _t10 = (_t4) * (_t9)
    _t11 = tl.tanh(_t10)
    _t12 = (_t3) + (_t11)
    _t13 = (_t2) * (_t12)
    mask_out = offsets < out_dim0
    tl.store(out_ptr + offsets, _t13, mask=mask_out)


def geluApprox(x, out):
    x_dim0 = x.shape[0]
    out_dim0 = out.shape[0]
    grid = (triton.cdiv(out_dim0, 1024),)
    _geluApprox_kernel[grid](x, out, x_dim0, out_dim0, BLOCK_SIZE=1024)
