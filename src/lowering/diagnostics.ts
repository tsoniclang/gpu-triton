import type { TargetDiagnostic } from "@tsonic/target-api";
import type { GpuSourceSpan } from "@tsonic/target-gpu";
import { tritonGpuBackendId } from "../capabilities/matrix.js";

export function tritonUnsupportedIrDiagnostic(input: {
  readonly capabilityId: string;
  readonly message: string;
  readonly kernelName: string;
  readonly span?: GpuSourceSpan;
}): TargetDiagnostic {
  return {
    code: "TRITON_UNSUPPORTED_IR",
    category: "error",
    source: "gpu-triton",
    message: input.message,
    ...(input.span === undefined ? {} : { sourceSpan: input.span }),
    evidence: [
      `target.capability=${input.capabilityId}`,
      `gpu.kernel=${input.kernelName}`,
      `gpu.backend=${tritonGpuBackendId}`,
    ],
  };
}
