# Worker Briefing

Build a backend plugin, not a standalone compiler.

## First Work Package

Start with T1 and T2 together:

- package shell,
- backend identity,
- capability table,
- structured Triton model,
- printer,
- vector add fixture lowering,
- architecture scanners.

This provides enough shape for review before reductions and matmul.

## Allowed References

- `../tsonic-gpu/.analysis/gpu-target-starter-kit-20260702-131129/`
- `../tsonic-python/.analysis/python-target-starter-kit-20260702-131129/`
- Triton public documentation for generated syntax.
- Existing Tsonic target packs for package shape.

Do not copy C# or Rust target semantics into Triton lowering.

## Stop Conditions

Stop and report if:

- GPU IR lacks information needed for correct Triton code,
- Python host artifact contract is missing,
- lowering requires TypeScript AST access,
- hardware is required for default tests,
- a generated output needs CPU fallback to pass.

## PR Review Checklist

- backend consumes GPU IR only,
- capability table is data,
- printer is deterministic,
- no Python target project ownership,
- no CPU fallback,
- fixture IR is test-only,
- every supported operation has rejection tests.

