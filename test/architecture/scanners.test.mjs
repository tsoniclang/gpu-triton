import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const sourceRoot = join(repositoryRoot, "src");

function collectFiles(root, extension) {
  const results = [];
  const visit = (directory) => {
    for (const entry of readdirSync(directory)) {
      const fullPath = join(directory, entry);
      if (statSync(fullPath).isDirectory()) {
        visit(fullPath);
        continue;
      }
      if (fullPath.endsWith(extension)) {
        results.push(fullPath);
      }
    }
  };
  visit(root);
  return results;
}

const sourceFiles = collectFiles(sourceRoot, ".ts").map((path) => ({
  path,
  text: readFileSync(path, "utf8"),
}));

test("the backend never consumes TypeScript ASTs or TSTS internals", () => {
  for (const { path, text } of sourceFiles) {
    assert.doesNotMatch(text, /@tsonic\/tsts/u, `${path} imports TSTS; the Triton backend consumes GPU IR only`);
  }
});

test("no tsonic-python or host-project ownership in the backend", () => {
  const banned = /tsonic-python|pyproject|writeFile|node:fs/iu;
  for (const { path, text } of sourceFiles) {
    assert.doesNotMatch(text, banned, `${path} reaches into host project ownership`);
  }
});

test("no CPU-recovery semantics in the backend", () => {
  for (const { path, text } of sourceFiles) {
    assert.doesNotMatch(text, /fallback/iu, `${path} mentions a recovery lane; Triton lowering must fail closed`);
  }
});

test("no product dependency on analysis files", () => {
  for (const { path, text } of sourceFiles) {
    assert.ok(!text.includes(".analysis/") && !text.includes('".analysis"'), `${path} references .analysis`);
  }
});

test("no runtime reflection, dynamic evaluation, or JS runtime imports", () => {
  const banned = /Reflect\.|\beval\(|new Function\(|node:child_process|node:http/u;
  for (const { path, text } of sourceFiles) {
    assert.doesNotMatch(text, banned, `${path} uses reflection or runtime facilities`);
  }
});

test("no PyTorch control flow in core lowering", () => {
  for (const { path, text } of sourceFiles) {
    assert.doesNotMatch(text, /torch/iu, `${path} references PyTorch in backend core`);
  }
});

test("lowering decides semantics from IR structure, not from generated text", () => {
  for (const { path, text } of sourceFiles) {
    if (!path.includes("/lowering/") && !path.includes("/backend/")) {
      continue;
    }
    if (path.endsWith("names.ts")) {
      continue;
    }
    assert.doesNotMatch(text, /\.match\(|\.replace\(|JSON\.parse\(/u, `${path} re-parses generated text`);
  }
});

test("lowering never emits artifacts alongside diagnostics", () => {
  const backendText = readFileSync(join(sourceRoot, "backend/triton-backend.ts"), "utf8");
  assert.match(backendText, /const diagnostics = this\.validate\(module\);\s*if \(diagnostics\.length > 0\) \{\s*throw new Error\(/u);
});

test("the capability matrix is data, not control flow", async () => {
  const { tritonCapabilities, tritonIntrinsicRows } = await import("../../dist/index.js");
  assert.ok(Object.isFrozen(tritonCapabilities));
  assert.ok(tritonCapabilities.capabilityIds.length > 20);
  assert.ok(tritonIntrinsicRows.every((row) => typeof row.tritonExpression === "string"));
});
