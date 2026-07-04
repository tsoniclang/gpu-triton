import { test } from "node:test";
import assert from "node:assert/strict";
import { createTritonGpuBackend } from "../dist/index.js";

// Naming policy: source-visible names are never auto-recased. Emitted
// identifiers and artifact paths are sanitized only for Python validity and
// path safety, and the raw IR kernel name stays available for correlation.

function mixedCaseModule() {
  return {
    name: "gpu_kernels",
    kernels: [
      {
        name: "MixedCase_Kernel9",
        parameters: [
          {
            kind: "tensor",
            name: "OutTensor",
            role: "output",
            tensor: {
              elementType: "float32",
              rank: 1,
              shape: [{ kind: "symbol", name: "OutTensor_dim0" }],
              layout: { kind: "contiguous" },
              device: { domain: "cuda" },
              mutability: "mutable",
              aliasing: "noalias",
            },
          },
        ],
        launch: { grid: [{ kind: "symbol", name: "OutTensor_dim0" }], streamPolicy: "default", devicePolicy: "single-device" },
        effects: [{ kind: "write", parameter: "OutTensor" }],
        body: {
          operations: [
            { kind: "thread-index", result: "i", space: "global", dimension: 0 },
            { kind: "const", result: "%1", dtype: "float32", value: 1 },
            { kind: "store", tensor: "OutTensor", indices: ["i"], value: "%1" },
          ],
        },
      },
    ],
  };
}

test("valid mixed-case names pass through byte-for-byte and stay correlatable", () => {
  const backend = createTritonGpuBackend();
  const module = mixedCaseModule();
  assert.deepEqual(backend.validate(module), []);
  const artifacts = backend.lower(module, { hostTargetId: "python" });
  const [emitted] = artifacts.modules;
  assert.equal(emitted.path, "kernels/MixedCase_Kernel9.py");
  assert.ok(emitted.text.includes("def MixedCase_Kernel9("));
  assert.ok(emitted.text.includes("def _MixedCase_Kernel9_kernel("));
  assert.ok(emitted.text.includes("OutTensor_ptr"));
  assert.deepEqual(artifacts.launchWrappers, [
    { hostFunctionName: "MixedCase_Kernel9", kernelName: "MixedCase_Kernel9", metaParameters: [] },
  ]);
});

test("sanitized kernels keep the raw IR name for correlation", () => {
  const backend = createTritonGpuBackend();
  const module = mixedCaseModule();
  module.kernels = [{ ...module.kernels[0], name: "Kernel With Spaces" }];
  assert.deepEqual(backend.validate(module), []);
  const artifacts = backend.lower(module, { hostTargetId: "python" });
  const [wrapper] = artifacts.launchWrappers;
  assert.equal(wrapper.kernelName, "Kernel With Spaces");
  assert.match(wrapper.hostFunctionName, /^[A-Za-z_][A-Za-z0-9_]*$/u);
  assert.equal(wrapper.hostFunctionName, "Kernel_With_Spaces");
  assert.equal(artifacts.modules[0].path, "kernels/Kernel_With_Spaces.py");
});
