import triton
import triton.language as tl


@triton.jit
def _total_kernel(values_ptr, out_ptr, values_dim0, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < values_dim0
    values_vals = tl.load(values_ptr + offsets, mask=mask, other=0.0)
    partial = tl.sum(values_vals)
    tl.store(out_ptr, partial)


def total(values, out):
    values_dim0 = values.shape[0]
    block_size = triton.next_power_of_2(values_dim0)
    grid = (1,)
    _total_kernel[grid](values, out, values_dim0, BLOCK_SIZE=block_size)
