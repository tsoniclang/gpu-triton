# GPU Triton Backend Starter Kit

This directory is the working specification for `gpu-triton`, the first concrete backend plugin for
the backend-neutral GPU target family.

`gpu-triton` lowers `tsonic-gpu` GPU IR into Triton/Python artifacts. It does not own TypeScript
checking, GPU IR design, Python project layout, PyTorch provider semantics, or JavaScript
compatibility.

## Read Order

1. `01-context-and-boundaries.md`
2. `02-architecture.md`
3. `03-triton-ir-lowering-contract.md`
4. `04-python-host-output-contract.md`
5. `05-capability-matrix.md`
6. `06-work-slices.md`
7. `07-tests-and-gates.md`
8. `08-user-code-examples.md`
9. `09-worker-briefing.md`

## One-Line Direction

Lower proven GPU IR to deterministic Triton/Python code, using normal Triton and Python library
APIs, without inventing GPU semantics in the backend.

## Non-Negotiables

- `gpu-triton` consumes GPU IR; it does not re-analyze TypeScript ASTs.
- `gpu-triton` does not define the tensor ABI; `tsonic-gpu` does.
- `gpu-triton` does not own Python project files; `tsonic-python` does.
- No CPU fallback.
- No PyTorch hardcoding in the backend core beyond imports required by generated host wrappers.
- No open dynamic dispatch or runtime reflection.
- Unsupported IR fails with deterministic diagnostics.

## Architecture Sketch

```text
 tsonic-gpu
  GPU IR + launch ABI + facts
          |
          v
 gpu-triton
  +--------------------------------+
  | capability check               |
  | IR to Triton AST/model         |
  | deterministic Python printer   |
  | host artifact contribution     |
  +--------------------------------+
          |
          v
 tsonic-python
  pyproject + modules + tests
```

## Completion Bar

The backend is useful when a vector-add kernel, a fused elementwise kernel, a reduction, and a
tiled matmul can be lowered from GPU IR into Triton code that passes syntax/import checks and runs
under an opt-in hardware gate.

