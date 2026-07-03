# Architecture

The Triton backend has five components:

```text
                 +-----------------------------+
                 | GPU IR from tsonic-gpu      |
                 +--------------+--------------+
                                |
                                v
 +-------------------+   +---------------------+   +--------------------+
 | capability table  |-->| IR lowering model   |-->| Triton printer     |
 +-------------------+   +---------------------+   +--------------------+
                                |
                                v
                 +-----------------------------+
                 | host artifact contribution  |
                 +-----------------------------+
                                |
                                v
                 +-----------------------------+
                 | tsonic-python package emit  |
                 +-----------------------------+
```

## Internal Modules

Suggested package shape:

```text
src/
  index.ts
  backend/
    triton-backend.ts
    capabilities.ts
    diagnostics.ts
  lowering/
    lower-module.ts
    lower-expression.ts
    lower-statement.ts
    lower-launch.ts
    lower-types.ts
  model/
    triton-module.ts
    triton-expression.ts
    triton-statement.ts
    triton-kernel.ts
  printer/
    python-printer.ts
    triton-printer.ts
  host/
    artifact-contribution.ts
test/
  backend-capabilities.test.mjs
  lowering-elementwise.test.mjs
  lowering-reduction.test.mjs
  lowering-matmul.test.mjs
  printer.test.mjs
  architecture.test.mjs
```

## No Stringly IR

Lowering must not concatenate arbitrary Python fragments as the semantic model. Use a structured
Triton/Python model and a printer.

Allowed:

```text
TritonCall(name: "tl.load", args: [...], kwargs: [...])
```

Rejected:

```text
"tl.load(" + ptr + " + " + index + ")"
```

## Diagnostics

Diagnostics must include:

- backend id `triton`,
- source span from GPU IR,
- GPU capability id,
- Triton reason,
- deterministic message.

Example:

```text
TRITON_UNSUPPORTED_GPU_IR
capability: gpu.atomic.float64
reason: selected Triton capability set does not support float64 atomic add
```

