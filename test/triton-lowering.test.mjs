import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { artifactText, compileGpu } from "./helpers/gpu-session.mjs";

// Golden Triton output tests. Goldens under test/golden/ are reviewed by
// hand; regenerate only after verifying the new lowering is correct, never
// to make a failing test pass.

const goldenRoot = join(dirname(fileURLToPath(import.meta.url)), "golden");

export const kernelSources = {
  add: `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";
import type { int32 } from "@tsonic/core/types.js";

export const add = kernel(function add(a: Float32Tensor, b: Float32Tensor, out: Float32Tensor, n: int32) {
  const i = gpu.globalId(0);
  if (i < n) {
    out[i] = a[i] + b[i];
  }
});
`,
  geluApprox: `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const geluApprox = kernel(function geluApprox(x: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  const v = x[i];
  out[i] = 0.5 * v * (1.0 + gpu.tanh(0.79788456 * (v + 0.044715 * v * v * v)));
});
`,
  total: `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const total = kernel(function total(values: Float32Tensor, out: Float32Tensor) {
  const partial = gpu.blockReduceSum(values);
  const lane = gpu.localId(0);
  if (lane === 0) {
    out[0] = partial;
  }
});
`,
  matmul: `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Matrix } from "@acme/tensor";

export const matmul = kernel(function matmul<M extends number, K extends number, N extends number>(
  a: Matrix<M, K>,
  b: Matrix<K, N>,
  c: Matrix<M, N>,
) {
  const row = gpu.globalId(0);
  const col = gpu.globalId(1);
  let acc = 0.0;
  const kDim = gpu.dim(a, 1);
  for (let k = 0; k < kDim; k++) {
    acc += a.at(row, k) * b.at(k, col);
  }
  c.set(row, col, acc);
});
`,
};

export function compileKernel(name) {
  return compileGpu({ files: { "index.ts": kernelSources[name] } });
}

for (const name of Object.keys(kernelSources)) {
  test(`golden Triton lowering: ${name}`, () => {
    const { result } = compileKernel(name);
    assert.deepEqual(result.diagnostics, []);
    const golden = readFileSync(join(goldenRoot, `${name}.py`), "utf8");
    assert.equal(artifactText(result, `kernels/${name}.py`), golden);
  });
}

test("launch plan requests the triton dependency and wrappers", () => {
  const { result } = compileKernel("add");
  const launchPlan = JSON.parse(artifactText(result, "gpu/launch-plan.json"));
  assert.equal(launchPlan.backend, "triton");
  assert.equal(launchPlan.hostTarget, "python");
  assert.deepEqual(launchPlan.dependencies, [{ ecosystem: "python", name: "triton" }]);
  assert.deepEqual(launchPlan.launchWrappers, [{ hostFunctionName: "add", kernelName: "add", metaParameters: [] }]);
});

test("artifact emission is deterministic", () => {
  const first = compileKernel("matmul").result;
  const second = compileKernel("matmul").result;
  assert.deepEqual(first.artifacts, second.artifacts);
});

test("matmul artifacts prove shape-symbol IR reached the backend", () => {
  const { result } = compileKernel("matmul");
  const text = artifactText(result, "kernels/matmul.py");
  assert.ok(text.includes("M = a.shape[0]"));
  assert.ok(text.includes("K = a.shape[1]"));
  assert.ok(text.includes("N = b.shape[1]"));
  assert.ok(text.includes("tl.dot(a_tile, b_tile)"));
});
