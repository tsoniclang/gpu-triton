# Work Slices

The Triton backend can start before the full GPU core exists by using fixture GPU IR. Fixture IR is
test input only and must match the public GPU IR once `tsonic-gpu` stabilizes.

## T1 — Package Shell, Capabilities, And Printer Model

Build:

- npm package shell,
- backend plugin identity,
- capability table,
- structured Python/Triton model,
- deterministic printer,
- architecture scanners.

Proof:

- backend describes capabilities,
- printer emits stable Python,
- no TSTS imports,
- no Python target project writes,
- unsupported fixture IR reports diagnostics.

## T2 — Elementwise Kernels

Build:

- vector load/store lowering,
- masks,
- block size meta parameters,
- 1D launch wrapper,
- dtype checks.

Proof:

- vector add fixture lowers,
- fused unary/binary expression lowers,
- unsupported dtype rejects,
- deterministic output snapshot.

## T3 — Reductions

Build:

- sum reduction lowering,
- identity values,
- output shape checks,
- launch wrapper for reductions.

Proof:

- sum fixture lowers,
- unsupported reduction op rejects,
- dtype mismatch rejects.

## T4 — Tiled Matrix Multiply

Build:

- 2D launch,
- tile offsets,
- dot product lowering,
- mask handling,
- meta parameter rows.

Proof:

- matmul fixture lowers,
- shape mismatch rejects,
- unsupported dtype rejects.

## T5 — Host Integration

Build:

- artifact contribution adapter for `tsonic-python`,
- dependency request rows,
- launch wrapper export metadata.

Proof:

- fake Python host receives artifacts,
- generated package shape is deterministic through host target tests.

## T6 — Hardware Gate

Build:

- opt-in hardware test harness,
- vector add run,
- reduction run,
- matmul run.

Proof:

- tests skip without explicit GPU marker,
- no CPU fallback,
- generated code runs when environment supports Triton.

## T7 — Final Backend Closure

Build:

- complete capability ledger,
- parity report against GPU core examples,
- diagnostics coverage,
- formatter/linter gates.

Proof:

- every supported row has positive and negative tests,
- every unsupported row is classified,
- generated code has no semantic string assembly.

