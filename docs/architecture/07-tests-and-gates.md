# Tests And Gates

`gpu-triton` tests must prove lowering correctness without requiring GPU hardware by default.

## Default Gates

- package build,
- TypeScript typecheck,
- printer snapshot tests,
- lowering tests from fixture GPU IR,
- capability rejection tests,
- architecture scanner tests.

## Architecture Scanner Must Reject

- imports from TSTS internals,
- imports from `tsonic-python` internals,
- writes to Python project files,
- CPU fallback code,
- string-based semantic lowering,
- `.analysis/` imports,
- PyTorch-specific control flow in core lowering,
- JS/Node runtime imports.

## Fixture IR Rules

Fixture IR is allowed until `tsonic-gpu` exports stable IR types. It must:

- be located under tests,
- match the intended public IR concepts,
- not become a second production IR schema,
- be deleted or replaced when public GPU IR package is available.

## Hardware Gate

Hardware tests are opt-in:

```text
TSONIC_GPU_HARDWARE=1 npm test -- gpu-triton-hardware
```

Default CI must not require CUDA hardware.

## Output Tests

Output tests should validate:

- stable imports,
- stable function names,
- deterministic formatting,
- no unused host artifacts,
- syntactic Python validity when Python is available,
- Triton import/run behavior in hardware gate.

