import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createTritonGpuBackend } from "../dist/index.js";

// The backend accepts any validate-clean GPU IR, so ids and kernel names it
// did not mint (a-b, leading digits, ../escape) must still produce valid
// Python identifiers and safe artifact paths.

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

function kernelNamed(name, scalarNames) {
  return {
    name,
    parameters: [tensor("out"), ...scalarNames.map((scalarName) => scalar(scalarName))],
    launch: { grid: [{ kind: "symbol", name: "out_dim0" }], streamPolicy: "default", devicePolicy: "single-device" },
    effects: [{ kind: "write", parameter: "out" }],
    body: {
      operations: [
        { kind: "thread-index", result: "i", space: "global", dimension: 0 },
        { kind: "const", result: "%1", dtype: "float32", value: 1 },
        { kind: "store", tensor: "out", indices: ["i"], value: "%1" },
      ],
    },
  };
}

const identifierPattern = /^[A-Za-z_][A-Za-z0-9_]*$/u;

function assertValidPython(text, label) {
  for (const [, name, parameters] of text.matchAll(/^def (\w+)\(([^)]*)\):$/gmu)) {
    const names = parameters
      .split(",")
      .map((parameter) => parameter.split(":")[0].trim())
      .filter((parameter) => parameter.length > 0);
    assert.equal(new Set(names).size, names.length, `${label}: duplicate parameters in ${name}`);
    for (const parameter of names) {
      assert.match(parameter, identifierPattern, `${label}: invalid parameter '${parameter}'`);
    }
  }
  for (const [, target] of text.matchAll(/^\s*([^=\s(]+) = /gmu)) {
    assert.match(target, identifierPattern, `${label}: invalid assignment target '${target}'`);
  }
  if (spawnSync("python3", ["--version"]).status === 0) {
    const parsed = spawnSync("python3", ["-c", "import ast, sys; ast.parse(sys.stdin.read())"], { input: text });
    assert.equal(parsed.status, 0, `${label}: python could not parse generated module:\n${parsed.stderr}`);
  }
}

test("hostile IR ids produce valid Python identifiers with bijection preserved", () => {
  const backend = createTritonGpuBackend();
  const kernel = kernelNamed("hostile", []);
  kernel.parameters = [
    tensor("out"),
    scalar("a-b"),
    scalar("a_b"),
    scalar("0col"),
    scalar("class"),
  ];
  kernel.body.operations.splice(1, 0, {
    kind: "binary",
    result: "sum-1",
    operator: "add",
    left: "a-b",
    right: "a_b",
    dtype: "int32",
  });
  const module = { name: "gpu_kernels", kernels: [kernel] };
  assert.deepEqual(backend.validate(module), []);
  const artifacts = backend.lower(module, { hostTargetId: "python" });
  const [emitted] = artifacts.modules;
  assertValidPython(emitted.text, "hostile ids");
  const defLine = emitted.text.match(/^def _hostile_kernel\(([^)]*)\):$/mu);
  assert.notEqual(defLine, null);
  const parameters = defLine[1].split(",").map((parameter) => parameter.split(":")[0].trim());
  // a-b and a_b sanitize to the same base but must allocate distinct names.
  const aLike = parameters.filter((parameter) => parameter === "a_b" || parameter === "a_b_v");
  assert.equal(aLike.length, 2);
  assert.ok(parameters.some((parameter) => parameter === "n0col"));
  assert.ok(parameters.some((parameter) => parameter === "class_v"));
});

test("kernel names never reach artifact paths unsanitized", () => {
  const backend = createTritonGpuBackend();
  const module = { name: "gpu_kernels", kernels: [kernelNamed("../escape", [])] };
  assert.deepEqual(backend.validate(module), []);
  const artifacts = backend.lower(module, { hostTargetId: "python" });
  const [emitted] = artifacts.modules;
  assert.match(emitted.path, /^kernels\/[A-Za-z0-9_]+\.py$/u, `unsafe artifact path: ${emitted.path}`);
  assert.ok(!emitted.path.includes(".."));
  assertValidPython(emitted.text, "escape kernel");
  assert.equal(artifacts.launchWrappers[0].hostFunctionName, emitted.path.slice("kernels/".length, -".py".length));
  assert.equal(artifacts.launchWrappers[0].kernelName, "../escape");
});

test("kernels whose names sanitize identically get distinct module paths", () => {
  const backend = createTritonGpuBackend();
  const module = { name: "gpu_kernels", kernels: [kernelNamed("a-b", []), kernelNamed("a_b", [])] };
  assert.deepEqual(backend.validate(module), []);
  const artifacts = backend.lower(module, { hostTargetId: "python" });
  const paths = artifacts.modules.map((emitted) => emitted.path);
  assert.equal(new Set(paths).size, paths.length, `colliding module paths: ${paths.join(", ")}`);
  const wrapperNames = artifacts.launchWrappers.map((wrapper) => wrapper.hostFunctionName);
  assert.equal(new Set(wrapperNames).size, wrapperNames.length);
  for (const emitted of artifacts.modules) {
    assertValidPython(emitted.text, emitted.path);
  }
});
