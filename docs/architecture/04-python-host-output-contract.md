# Python Host Output Contract

`gpu-triton` contributes Python artifacts; `tsonic-python` places them in a package.

## Artifact Contribution

A contribution should contain:

```text
module path request
imports
dependency requirements
kernel functions
launch wrappers
test/run metadata
```

Example:

```text
dependency:
  name: triton
module:
  relativePath: kernels/add.py
exports:
  add_kernel
  add
```

## Generated Module Shape

For a vector add kernel, generated Python should look structurally like:

```py
import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)

def add(a, b, out, n: int):
    grid = (triton.cdiv(n, 1024),)
    _add_kernel[grid](a, b, out, n, BLOCK_SIZE=1024)
```

The final printer must produce deterministic formatting. The model decides semantics; formatting is
only presentation.

## Host Dependencies

The backend can request:

- `triton`,
- `torch` for wrapper examples or torch tensor launch paths.

If a user provides a non-PyTorch tensor library with compatible pointer/shape facts, the GPU core
should still be able to produce IR. Backend wrapper generation may require a library-specific host
adapter supplied through Python provider metadata.

## No Python Target Ownership

The backend must not:

- write `pyproject.toml`,
- choose source layout,
- run Python package installers,
- own CLI commands.

It returns artifact requests to `tsonic-python`.

