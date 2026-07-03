# Context And Boundaries

Triton is the first GPU backend because it gives a practical bridge between Python ML ecosystems and
high-performance generated kernels. The backend must stay narrow: it receives GPU IR and emits
Triton artifacts.

## What This Repository Owns

- Triton backend package identity.
- Backend capability declaration.
- GPU IR validation against Triton capability rows.
- Lowering from GPU IR to Triton operation model.
- Deterministic Python/Triton printer.
- Host artifact contribution records for `tsonic-python`.
- Triton-focused tests and diagnostics.

## What This Repository Does Not Own

- TypeScript AST analysis.
- TSTS extension APIs.
- GPU IR schema.
- Python target pack.
- PyTorch provider package semantics.
- JavaScript compatibility.
- Node compatibility.
- CPU fallback.
- Hardware discovery policy.

## Dependency Policy

Generated output can depend on normal Python packages:

- `triton`,
- `torch` when host wrappers need tensor launch examples,
- standard Python library modules.

The backend product code should avoid unnecessary runtime dependencies. Most tests can validate
printed artifacts without importing Triton.

## Backend Selection

`gpu-triton` is selected by the GPU target through the installed-plugin contract. The package
declares the core host manifest in package.json:

```json
"tsonic": { "kind": "plugin", "contractVersion": 1, "entry": "." }
```

and exports `createTsonicPlugin()`, which returns a plugin object carrying
`kind: "gpu-backend"`, `backendId: "triton"`, and `createBackend()`. Plugin kinds live on
returned objects, never in package.json metadata. The GPU target composes gpu-backend and
gpu-host plugin entries (core host routing is a tsonic core requirement; local composition
passes them to `createTsonicPlugin({ plugins })`), and selection comes from target options:

```json
{ "id": "gpu", "options": { "backendId": "triton", "hostTargetId": "python" } }
```

The backend does not self-select when it sees `torch` or `triton` imports.

