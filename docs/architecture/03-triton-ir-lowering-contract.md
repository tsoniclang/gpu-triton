# Triton IR Lowering Contract

The backend receives validated GPU IR. It must still check that the selected Triton capability set
can lower every operation.

## Lowering Inputs

```text
GpuIrModule
GpuLaunchPlan
GpuBackendCapabilitySet
HostArtifactContext
```

The backend must not require TypeScript AST nodes.

## Lowering Outputs

```text
TritonBackendArtifacts
  modules
  imports
  dependencies
  launch wrappers
  diagnostics
```

## Triton Model

The backend should define a small structured model:

- `PyModule`
- `PyImport`
- `PyFunction`
- `PyAssignment`
- `PyReturn`
- `PyCall`
- `PySubscript`
- `PyBinary`
- `TritonKernel`
- `TritonProgramId`
- `TritonLoad`
- `TritonStore`
- `TritonArange`
- `TritonMask`

The printer converts that model into Python source.

## Elementwise Lowering

GPU IR:

```text
i = global_id(0)
mask = i < N
out[i] = a[i] + b[i]
```

Triton model:

```text
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n
a_values = tl.load(a_ptr + offsets, mask=mask)
b_values = tl.load(b_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, a_values + b_values, mask=mask)
```

## Reduction Lowering

Reductions require explicit backend capability. The backend must know:

- reduction axis,
- block size,
- identity value,
- operation,
- output shape.

Unsupported reductions fail before printing.

## Matrix Multiply Lowering

Matmul lowering should be a distinct path with explicit constraints:

- ranks are 2,
- dtypes supported,
- shapes `[M, K] x [K, N] -> [M, N]`,
- layout supported,
- block sizes selected from backend policy rows.

Tiling policy belongs in `gpu-triton`, not `tsonic-gpu`, because it is backend-specific.

## Intrinsics

Map GPU math intrinsics to Triton equivalents through declarative rows:

```text
gpu.tanh -> tl.tanh
gpu.exp  -> tl.exp
gpu.log  -> tl.log
```

Rows are data. Lowering logic branches on intrinsic capability kind, not source spelling.

