import { test } from "node:test";
import assert from "node:assert/strict";
import { createTsonicPlugin as createGpuTargetTsonicPlugin, createFakeGpuHostIntegration } from "@tsonic/target-gpu";
import { createTsonicPlugin, readTsonicPluginManifest, tritonBackendPluginId } from "../dist/index.js";
import { artifactText, compileGpu } from "./helpers/gpu-session.mjs";

test("the tsonic manifest follows the core plugin contract", () => {
  assert.deepEqual(readTsonicPluginManifest(), { kind: "plugin", contractVersion: 1, entry: "." });
});

test("package.json resolves through package exports for host discovery", async () => {
  const { createRequire } = await import("node:module");
  const requireFromHere = createRequire(import.meta.url);
  const resolved = requireFromHere.resolve("@tsonic/gpu-triton/package.json");
  assert.ok(resolved.endsWith("package.json"));
});

test("createTsonicPlugin exposes Triton as one GPU backend plugin", () => {
  const plugin = createTsonicPlugin();
  assert.equal(plugin.kind, "gpu-backend");
  assert.equal(plugin.id, tritonBackendPluginId);
  assert.equal(plugin.backendId, "triton");
  assert.equal(plugin.createBackend().id, "triton");
});

test("the GPU target composes the Triton plugin end-to-end", () => {
  const targetPlugin = createGpuTargetTsonicPlugin({
    plugins: [
      createTsonicPlugin(),
      { kind: "gpu-host", id: "@fake/python-host", hostTargetId: "python", createHostIntegration: () => createFakeGpuHostIntegration("python") },
    ],
  });
  const pack = targetPlugin.createTargetPack();
  const source = `import { kernel, gpu } from "@tsonic/gpu/lang.js";
import type { Float32Tensor } from "@acme/tensor";

export const scale = kernel(function scale(a: Float32Tensor, out: Float32Tensor) {
  const i = gpu.globalId(0);
  out[i] = a[i] * 2.0;
});
`;
  const { result } = compileGpu({ files: { "index.ts": source }, pack });
  assert.deepEqual(result.diagnostics, []);
  const text = artifactText(result, "kernels/scale.py");
  assert.ok(text.includes("@triton.jit"));
  const launchPlan = JSON.parse(artifactText(result, "gpu/launch-plan.json"));
  assert.equal(launchPlan.backend, "triton");
});
