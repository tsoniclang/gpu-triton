import { test } from "node:test";
import assert from "node:assert/strict";
import { createTritonGpuBackend } from "../dist/index.js";

// Regression for collision-safe Python naming: distinct IR ids must never
// map to the same Python identifier in one kernel. `lambda` sanitizes to
// `lambda_v`, which must not collide with an actual `lambda_v` id.

function tensor(name) {
  return {
    kind: "tensor",
    name,
    role: "output",
    tensor: {
      elementType: "float32",
      rank: 1,
      shape: [{ kind: "symbol", name: `${name}_dim0` }],
      layout: { kind: "contiguous" },
      device: { domain: "cuda" },
      mutability: "mutable",
      aliasing: "noalias",
    },
  };
}

function scalar(name) {
  return { kind: "scalar", name, role: "scalar", scalarType: "int32" };
}

function collisionModule() {
  return {
    name: "gpu_kernels",
    kernels: [
      {
        name: "collision",
        parameters: [
          { ...tensor("out"), role: "output" },
          scalar("lambda"),
          scalar("lambda_v"),
          scalar("mask_out"),
        ],
        launch: { grid: [{ kind: "symbol", name: "out_dim0" }], streamPolicy: "default", devicePolicy: "single-device" },
        effects: [{ kind: "write", parameter: "out" }],
        body: {
          operations: [
            { kind: "thread-index", result: "i", space: "global", dimension: 0 },
            { kind: "binary", result: "%1", operator: "add", left: "lambda", right: "lambda_v", dtype: "int32" },
            { kind: "binary", result: "%2", operator: "add", left: "%1", right: "mask_out", dtype: "int32" },
            { kind: "binary", result: "%3", operator: "lt", left: "i", right: "%2", dtype: "int32" },
            { kind: "const", result: "%4", dtype: "float32", value: 1 },
            {
              kind: "if",
              condition: "%3",
              then: { operations: [{ kind: "store", tensor: "out", indices: ["i"], value: "%4" }] },
            },
          ],
        },
      },
    ],
  };
}

function parameterLists(text) {
  return [...text.matchAll(/^def (\w+)\(([^)]*)\):$/gmu)].map(([, name, parameters]) => [
    name,
    parameters
      .split(",")
      .map((parameter) => parameter.split(":")[0].trim())
      .filter((parameter) => parameter.length > 0),
  ]);
}

test("colliding IR ids allocate distinct Python names", () => {
  const backend = createTritonGpuBackend();
  const module = collisionModule();
  assert.deepEqual(backend.validate(module), []);
  const artifacts = backend.lower(module, { hostTargetId: "python" });
  const [emitted] = artifacts.modules;
  assert.notEqual(emitted, undefined);
  const functions = parameterLists(emitted.text);
  assert.ok(functions.length >= 2);
  for (const [name, parameters] of functions) {
    assert.equal(new Set(parameters).size, parameters.length, `duplicate parameters in ${name}: ${parameters.join(", ")}`);
  }
  // The sanitized `lambda` and the literal `lambda_v` must be two names.
  const kernelParameters = functions[0][1];
  const lambdaLike = kernelParameters.filter((parameter) => parameter.startsWith("lambda"));
  assert.equal(new Set(lambdaLike).size, 2, `expected two distinct lambda-derived names: ${lambdaLike.join(", ")}`);
  // The emitted bounds mask must not collide with the scalar named mask_out.
  const assignedNames = [...emitted.text.matchAll(/^\s*(\w+) = /gmu)].map(([, name]) => name);
  const maskAssignments = assignedNames.filter((name) => name.startsWith("mask_out"));
  assert.ok(maskAssignments.length > 0, "expected an emitted bounds mask for tensor 'out'");
  assert.ok(
    maskAssignments.every((name) => !kernelParameters.includes(name)),
    `bounds mask collides with a kernel parameter: ${maskAssignments.join(", ")}`,
  );
});

test("sanitized kernel names flow into launch wrappers", () => {
  const backend = createTritonGpuBackend();
  const module = collisionModule();
  const artifacts = backend.lower(module, { hostTargetId: "python" });
  assert.deepEqual(artifacts.launchWrappers, [
    { hostFunctionName: "collision", kernelName: "collision", metaParameters: [] },
  ]);
});
