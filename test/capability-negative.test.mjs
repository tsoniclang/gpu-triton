import { test } from "node:test";
import assert from "node:assert/strict";
import { createTritonGpuBackend } from "../dist/index.js";
import { capabilityIds, compileGpu } from "./helpers/gpu-session.mjs";

// Every unsupported lane must fail with deterministic diagnostics and zero
// artifacts. There is no CPU path and no partial lowering.

function assertFailsClosed(result, capability) {
  assert.deepEqual(result.artifacts, []);
  assert.ok(result.diagnostics.length > 0);
  assert.ok(
    result.diagnostics.every((diagnostic) => diagnostic.category === "error"),
    "all diagnostics must be errors",
  );
  assert.ok(
    capabilityIds(result.diagnostics).includes(capability),
    `expected capability '${capability}'; found: ${capabilityIds(result.diagnostics).join(", ")}`,
  );
}

test("unsupported dtype fails closed before lowering", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float64Tensor } from "@acme/tensor";

export const wide = kernel(function wide(a: Float64Tensor, out: Float64Tensor) {
  const i = gpu.globalId(0);
  out[i] = a[i];
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assertFailsClosed(result, "gpu.dtype.float64");
});

test("unsupported reduction operator fails closed", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const peak = kernel(function peak(values: Float32Tensor, out: Float32Tensor) {
  const partial = gpu.blockReduceMax(values);
  const lane = gpu.localId(0);
  if (lane === 0) {
    out[0] = partial;
  }
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assertFailsClosed(result, "gpu.reduce.max.float32");
});

test("rank-2 kernels outside the matmul pattern fail closed", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Matrix } from "@acme/tensor";

export const diag = kernel(function diag<M extends number, N extends number>(a: Matrix<M, N>, c: Matrix<M, N>) {
  const i = gpu.globalId(0);
  c.set(i, i, a.at(i, i));
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assertFailsClosed(result, "triton.lowering.kernel-shape");
  assert.ok(result.diagnostics.some((diagnostic) => diagnostic.code === "TRITON_UNSUPPORTED_IR"));
});

test("meta-indexed kernels fail closed on the launch meta lane", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const tiled = kernel(function tiled(a: Float32Tensor, out: Float32Tensor) {
  const block = gpu.meta("BLOCK");
  out[block] = a[block];
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assertFailsClosed(result, "triton.lowering.launch-meta");
});

test("serial per-element loops have no Triton lowering", () => {
  const source = `import { kernel } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";
import type { int32 } from "@tsonic/core/types.js";

export const copy = kernel(function copy(a: Float32Tensor, out: Float32Tensor, n: int32) {
  for (let k = 0; k < n; k++) {
    out[k] = a[k];
  }
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assertFailsClosed(result, "triton.lowering.kernel-shape");
});

test("invalid tensor shape binding fails closed in the GPU core", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Matrix, Float32Tensor } from "@acme/tensor";

export const bad = kernel(function bad(a: Matrix<number, number>, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = a.at(i, i);
});
`;
  const { result } = compileGpu({ files: { "index.ts": source } });
  assertFailsClosed(result, "gpu.kernel.shape-symbols");
});

test("missing host integration fails closed after backend lowering is ready", () => {
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const scale = kernel(function scale(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = a[i] * 2.0;
});
`;
  const { result } = compileGpu({ files: { "index.ts": source }, hosts: [] });
  assertFailsClosed(result, "gpu.host.integration");
  assert.equal(result.diagnostics[0].code, "GPU_HOST_INTEGRATION_MISSING");
});

test("hand-authored IR beyond the capability matrix rejects", () => {
  const backend = createTritonGpuBackend();
  const tensor = (name, role, rank) => ({
    kind: "tensor",
    name,
    role,
    tensor: {
      elementType: "float32",
      rank,
      shape: Array.from({ length: rank }, (_, dimension) => ({ kind: "symbol", name: `${name}_dim${dimension}` })),
      layout: { kind: "contiguous" },
      device: { domain: "cuda" },
      mutability: role === "input" ? "readonly" : "mutable",
      aliasing: "noalias",
    },
  });
  const rank3 = {
    name: "gpu_kernels",
    kernels: [
      {
        name: "cube",
        parameters: [tensor("a", "input", 3), tensor("out", "output", 1)],
        launch: { grid: [{ kind: "symbol", name: "out_dim0" }], streamPolicy: "default", devicePolicy: "single-device" },
        effects: [
          { kind: "read", parameter: "a" },
          { kind: "write", parameter: "out" },
        ],
        body: { operations: [] },
      },
    ],
  };
  const rank3Ids = backend.validate(rank3).flatMap((diagnostic) =>
    (diagnostic.evidence ?? []).filter((row) => row.startsWith("target.capability=")).map((row) => row.slice(18)),
  );
  assert.ok(rank3Ids.includes("gpu.tensor.rank.3"));

  const atomicModule = {
    name: "gpu_kernels",
    kernels: [
      {
        name: "bump",
        parameters: [tensor("out", "output", 1)],
        launch: { grid: [{ kind: "symbol", name: "out_dim0" }], streamPolicy: "default", devicePolicy: "single-device" },
        effects: [{ kind: "write", parameter: "out" }],
        body: {
          operations: [
            { kind: "thread-index", result: "i", space: "global", dimension: 0 },
            { kind: "const", result: "one", dtype: "float32", value: 1 },
            { kind: "atomic", operator: "add", tensor: "out", indices: ["i"], value: "one", dtype: "float32" },
          ],
        },
      },
    ],
  };
  const atomicIds = backend.validate(atomicModule).flatMap((diagnostic) =>
    (diagnostic.evidence ?? []).filter((row) => row.startsWith("target.capability=")).map((row) => row.slice(18)),
  );
  assert.ok(atomicIds.includes("gpu.atomic.add.float32"));
});

test("lower refuses modules that fail validation", () => {
  const backend = createTritonGpuBackend();
  const invalid = { name: "gpu_kernels", kernels: [{ name: "", parameters: [], launch: { grid: [], streamPolicy: "default", devicePolicy: "single-device" }, effects: [], body: { operations: [] } }] };
  assert.throws(() => backend.lower(invalid, { hostTargetId: "python" }), /cannot lower module/u);
});

test("the capability matrix stays narrow where the spec defers", () => {
  const capabilities = createTritonGpuBackend().describeCapabilities();
  const deferred = capabilities.capabilityIds.filter(
    (id) =>
      id.startsWith("gpu.atomic.") ||
      id.startsWith("gpu.barrier.") ||
      id === "gpu.dtype.float64" ||
      id === "gpu.dtype.float16" ||
      id === "gpu.dtype.bfloat16" ||
      id.startsWith("gpu.reduce.max"),
  );
  assert.deepEqual(deferred, []);
});
