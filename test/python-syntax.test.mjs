import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const temporaryRoot = join(dirname(fileURLToPath(import.meta.url)), "..", ".temp");
mkdirSync(temporaryRoot, { recursive: true });
import { artifactText, compileGpu } from "./helpers/gpu-session.mjs";
import { kernelSources } from "./triton-lowering.test.mjs";

// Generated artifact gates: every emitted Python module must parse (ast) and
// byte-compile. Importing triton itself is a dependency-gated check: it runs
// when the environment provides triton and is skipped otherwise, mirroring
// the hardware gate rule — absence of the dependency never weakens the
// syntax gates above it.

function pythonAvailable() {
  return spawnSync("python3", ["--version"]).status === 0;
}

const havePython = pythonAvailable();
const haveTriton = havePython && spawnSync("python3", ["-c", "import triton"]).status === 0;

test("generated Triton modules parse and byte-compile", { skip: !havePython }, () => {
  const workDirectory = mkdtempSync(join(temporaryRoot, "syntax-"));
  for (const [name, source] of Object.entries(kernelSources)) {
    const { result } = compileGpu({ files: { "index.ts": source } });
    assert.deepEqual(result.diagnostics, []);
    const text = artifactText(result, `kernels/${name}.py`);
    const path = join(workDirectory, `${name}.py`);
    writeFileSync(path, text);
    execFileSync("python3", ["-c", "import ast, sys; ast.parse(open(sys.argv[1]).read())", path]);
    execFileSync("python3", ["-m", "py_compile", path]);
  }
});

test("generated Triton modules import against a real triton installation", { skip: !haveTriton }, () => {
  const workDirectory = mkdtempSync(join(temporaryRoot, "import-"));
  for (const [name, source] of Object.entries(kernelSources)) {
    const { result } = compileGpu({ files: { "index.ts": source } });
    const path = join(workDirectory, `${name}.py`);
    writeFileSync(path, artifactText(result, `kernels/${name}.py`));
    execFileSync("python3", [
      "-c",
      "import importlib.util, sys; spec = importlib.util.spec_from_file_location('m', sys.argv[1]); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)",
      path,
    ]);
  }
});
