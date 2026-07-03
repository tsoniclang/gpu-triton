import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { GpuBackendTsonicPlugin } from "@tsonic/target-gpu";
import { createTritonGpuBackend } from "./backend/triton-backend.js";
import { tritonGpuBackendId } from "./capabilities/matrix.js";

export const tritonBackendPluginId = "@tsonic/gpu-triton";

// The tsonic host plugin contract: discovery reads the package.json
// 'tsonic' manifest (kind 'plugin', contractVersion 1, entry resolved
// through package exports) and calls createTsonicPlugin(). The manifest
// shape is owned by tsonic core; this package follows it exactly. Routing
// the returned gpu-backend entry into @tsonic/target-gpu is a tsonic core
// responsibility (see tsonic-gpu docs/core-host-requests.md).
export interface TritonBackendPluginManifest {
  readonly kind: "plugin";
  readonly contractVersion: 1;
  readonly entry: string;
}

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export function readTsonicPluginManifest(): TritonBackendPluginManifest {
  const packageJsonPath = resolve(packageRoot, "package.json");
  const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8")) as {
    readonly tsonic?: unknown;
  };
  const manifest = packageJson.tsonic;
  if (manifest === undefined || typeof manifest !== "object" || manifest === null) {
    throw new Error(`${tritonBackendPluginId}: package.json is missing the 'tsonic' plugin manifest.`);
  }
  const { kind, contractVersion, entry } = manifest as {
    readonly kind?: unknown;
    readonly contractVersion?: unknown;
    readonly entry?: unknown;
  };
  if (kind !== "plugin") {
    throw new Error(`${tritonBackendPluginId}: tsonic manifest kind must be 'plugin', got '${String(kind)}'.`);
  }
  if (contractVersion !== 1) {
    throw new Error(`${tritonBackendPluginId}: tsonic manifest contractVersion must be 1, got '${String(contractVersion)}'.`);
  }
  if (typeof entry !== "string" || entry.length === 0) {
    throw new Error(`${tritonBackendPluginId}: tsonic manifest entry must be a non-empty string.`);
  }
  return { kind, contractVersion, entry };
}

// The installed-plugin entry: Triton is one GPU backend plugin among many,
// composed by the GPU target through data, never special compiler knowledge.
export function createTsonicPlugin(): GpuBackendTsonicPlugin {
  readTsonicPluginManifest();
  return {
    kind: "gpu-backend",
    id: tritonBackendPluginId,
    backendId: tritonGpuBackendId,
    createBackend: createTritonGpuBackend,
  };
}
