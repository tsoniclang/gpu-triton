# Capability Matrix

This matrix drives lowering. The backend must expose it as data.

## Supported Capability Matrix

| Area | Supported Rows |
| --- | --- |
| Dtypes | `float32`, `int32`, `bool` |
| Ranks | 1D elementwise, 2D matmul |
| Layouts | contiguous |
| Devices | CUDA-compatible Triton target |
| Arithmetic | add, sub, mul, div, neg |
| Comparison | lt, le, gt, ge, eq, ne |
| Math | exp, log, tanh, sqrt where Triton supports it |
| Memory | load, store with mask |
| Launch | 1D and 2D grid |
| Reduction | sum for supported dtype |
| Matrix | tiled matmul for float32 |

## Capabilities Outside The Matrix

| Area | Reason |
| --- | --- |
| float16/bfloat16 | Requires precision policy and tests |
| atomics | Requires explicit memory effect model |
| shared memory details | Triton-specific tiling policy needed |
| dynamic rank | Tensor ABI and backend constraints needed |
| irregular control flow | Legality and lowering proof needed |
| sparse tensors | Host tensor library semantics needed |

## Hard Reject

- host object calls inside kernels,
- Python exceptions inside kernels,
- dynamic property access,
- string operations,
- JS compat operations,
- Node operations,
- runtime reflection,
- CPU fallback.

## Capability Row Shape

```text
CapabilityRow
  id
  gpuOperationKind
  dtypeConstraints
  rankConstraints
  layoutConstraints
  loweringKind
  diagnosticWhenMissing
```

Rows are product data. Lowering code branches on `loweringKind`, not source API names.

