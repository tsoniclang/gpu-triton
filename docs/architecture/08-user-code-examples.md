# User Code Examples

These examples show what `gpu-triton` must lower after `tsonic-gpu` extracts GPU IR.

## Vector Add

Input TypeScript:

```ts
export const add = kernel(function add(a: Tensor<float32, [N]>, b: Tensor<float32, [N]>, out: Tensor<float32, [N]>) {
  const i = gpu.globalId(0);
  if (i < N) {
    out[i] = a[i] + b[i];
  }
});
```

Triton output shape:

```py
@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)
```

## Fused Elementwise

Input TypeScript:

```ts
export const fused = kernel(function fused(x: Tensor<float32, [N]>, out: Tensor<float32, [N]>) {
  const i = gpu.globalId(0);
  const v = x[i];
  out[i] = gpu.tanh(v) * v + 1.0;
});
```

Backend capabilities:

- `gpu.math.tanh.float32`,
- `gpu.arithmetic.mul.float32`,
- `gpu.arithmetic.add.float32`,
- masked load/store.

## Sum Reduction

Input TypeScript:

```ts
export const sum = kernel(function sum(values: Tensor<float32, [N]>, out: Tensor<float32, [1]>) {
  out[0] = gpu.sum(values);
});
```

The backend chooses a reduction lowering only if capability rows allow it.

## Matrix Multiply

Input TypeScript:

```ts
export const matmul = kernel(function matmul(
  a: Tensor<float32, [M, K]>,
  b: Tensor<float32, [K, N]>,
  c: Tensor<float32, [M, N]>,
) {
  const row = gpu.globalId(0);
  const col = gpu.globalId(1);
  let acc = 0.0;
  for (let k = 0; k < K; k++) {
    acc += a[row, k] * b[k, col];
  }
  c[row, col] = acc;
});
```

The Triton backend can lower this to a tiled matmul kernel. Tiling values are backend policy rows,
not GPU core policy.

## Hard Reject

Input TypeScript:

```ts
export const bad = kernel(function bad(x: Tensor<float32, [N]>, out: Tensor<float32, [N]>) {
  const i = gpu.globalId(0);
  out[i] = Math.random() * x[i];
});
```

If `Math.random` has no GPU intrinsic capability row, the backend rejects it. It must not emit a
host random call.

