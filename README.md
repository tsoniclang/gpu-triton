# gpu-triton

Triton backend plugin for the Tsonic GPU target family, published as `@tsonic/gpu-triton`.

`gpu-triton` implements the `GpuBackendPlugin` contract from `@tsonic/target-gpu`: it consumes
validated, backend-neutral GPU IR and lowers it to deterministic Triton/Python modules. It owns
no TypeScript analysis, no GPU IR design, and no Python project layout — the GPU core proves
legality once, and the selected host target places every artifact.

## Principles

- The backend consumes GPU IR; it never touches TypeScript ASTs or TSTS.
- The capability matrix is data (`src/capabilities/matrix.ts`); lowering branches on IR
  structure and capability rows, never on source spellings or generated text.
- Every kernel classifies into a supported lowering path before any code is produced;
  anything else is a deterministic `TRITON_UNSUPPORTED_IR` diagnostic and zero artifacts.
- There is no CPU path and no partial lowering.

## Installed Plugin Shape

The package declares a `tsonic` manifest (`kind: "gpu-backend"`, `target: "gpu"`,
`backend: "triton"`) in package.json and exports `createTsonicPlugin()`, which returns the
GPU backend plugin entry the GPU target composes. Triton is one backend plugin among many,
selected through target options — never special compiler knowledge.

## Lowering Paths

- **Elementwise (rank 1)** — the global thread index becomes a `BLOCK_SIZE`-wide offsets
  vector, scalar IR values become Triton vectors, and conditional guards become load/store
  masks alongside per-tensor bounds masks.
- **Block reduction** — whole-tensor sum lowered as a single-program masked block reduction
  with the operation identity.
- **Tiled matmul** — the recognized `[M,K] x [K,N] -> [M,N]` accumulator pattern lowers through
  the canonical 2D-grid `tl.dot` kernel; block sizes come from backend policy rows.

## Build and Test

The sibling `../tsonic` and `../tsonic-gpu` repositories must be built first; this repository
never builds or writes into them.

```sh
npm install
npm test
```

The default suite requires no GPU hardware and no Triton installation: golden output tests,
capability rejection tests, and Python syntax gates (`ast.parse`, `py_compile`) cover the
generated modules. The import gate runs only where the environment provides `triton`.
