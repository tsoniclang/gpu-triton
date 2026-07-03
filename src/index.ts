export { createTritonGpuBackend } from "./backend/triton-backend.js";
export {
  createTsonicPlugin,
  readTsonicPluginManifest,
  tritonBackendPluginId,
  type TritonBackendPluginManifest,
} from "./plugin.js";
export {
  tritonCapabilities,
  tritonElementwiseBlockSize,
  tritonGpuBackendId,
  tritonIntrinsicRows,
  tritonMatmulBlockPolicy,
  type TritonIntrinsicRow,
} from "./capabilities/matrix.js";
export {
  classifyTritonKernel,
  type ClassificationFailure,
  type ClassificationResult,
  type ElementwisePlan,
  type MatmulPlan,
  type ReductionPlan,
  type TritonKernelPlan,
} from "./lowering/classify.js";
export { tritonUnsupportedIrDiagnostic } from "./lowering/diagnostics.js";
export { createPyNameAllocator, pyName, type PyNameAllocator } from "./lowering/names.js";
export { printPyModule } from "./py/printer.js";
export { type PyFunction, type PyModule, type PyStatement } from "./py/model.js";
