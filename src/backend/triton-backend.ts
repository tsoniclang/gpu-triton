import type { TargetDiagnostic } from "@tsonic/target-api";
import {
  matchGpuModuleAgainstCapabilities,
  validateGpuIrModule,
  type GpuBackendArtifacts,
  type GpuBackendPlugin,
  type GpuHostLoweringContext,
  type GpuIrModule,
} from "@tsonic/target-gpu";
import { tritonCapabilities, tritonGpuBackendId } from "../capabilities/matrix.js";
import { classifyTritonKernel } from "../lowering/classify.js";
import { tritonUnsupportedIrDiagnostic } from "../lowering/diagnostics.js";
import { lowerElementwiseKernel } from "../lowering/elementwise.js";
import { lowerMatmulKernel } from "../lowering/matmul.js";
import { lowerReductionKernel } from "../lowering/reduction.js";
import { createPyNameAllocator } from "../lowering/names.js";
import { printPyModule } from "../py/printer.js";
import type { PyFunction } from "../py/model.js";

const tritonModuleImports = Object.freeze(["import triton", "import triton.language as tl"]);

// The Triton backend consumes validated GPU IR and emits deterministic
// Triton/Python modules. Any kernel outside the supported lowering paths is
// a deterministic diagnostic and zero artifacts — never a partial lowering
// and never a CPU path.
export function createTritonGpuBackend(): GpuBackendPlugin {
  return {
    id: tritonGpuBackendId,
    describeCapabilities() {
      return tritonCapabilities;
    },
    validate(module: GpuIrModule): readonly TargetDiagnostic[] {
      const diagnostics: TargetDiagnostic[] = [
        ...validateGpuIrModule(module),
        ...matchGpuModuleAgainstCapabilities(module, tritonCapabilities),
      ];
      if (diagnostics.length > 0) {
        return diagnostics;
      }
      for (const kernel of module.kernels) {
        const classification = classifyTritonKernel(kernel);
        if ("failure" in classification) {
          diagnostics.push(
            tritonUnsupportedIrDiagnostic({
              capabilityId: classification.failure.capabilityId,
              message: classification.failure.message,
              kernelName: kernel.name,
              ...(kernel.span === undefined ? {} : { span: kernel.span }),
            }),
          );
        }
      }
      return diagnostics;
    },
    lower(module: GpuIrModule, context: GpuHostLoweringContext): GpuBackendArtifacts {
      const diagnostics = this.validate(module);
      if (diagnostics.length > 0) {
        throw new Error(
          `The Triton backend cannot lower module '${module.name}': validation reported ${diagnostics.length} diagnostic(s).`,
        );
      }
      void context;
      const kernels = [...module.kernels].sort((left, right) => left.name.localeCompare(right.name, "en"));
      // Module names and artifact paths come only from allocated stems: one
      // bijective, sanitized identifier per kernel, never the raw IR name.
      const moduleNames = createPyNameAllocator();
      const stems = new Map(kernels.map((kernel) => [kernel.name, moduleNames.nameFor(kernel.name)]));
      const stemOf = (kernelName: string): string => stems.get(kernelName) ?? "kernel";
      const modules = kernels.map((kernel) => {
        const classification = classifyTritonKernel(kernel);
        if (!("plan" in classification)) {
          throw new Error(`The Triton backend cannot lower kernel '${kernel.name}' after validation accepted it.`);
        }
        const { plan } = classification;
        const functionStem = stemOf(kernel.name);
        const emitted: { kernelFunction: PyFunction; wrapperFunction: PyFunction } =
          plan.kind === "matmul"
            ? lowerMatmulKernel(kernel, plan, functionStem)
            : plan.kind === "reduction"
              ? lowerReductionKernel(kernel, plan, functionStem)
              : lowerElementwiseKernel(kernel, plan, functionStem);
        return {
          path: `kernels/${functionStem}.py`,
          language: "python",
          text: printPyModule({
            imports: [...tritonModuleImports],
            functions: [emitted.kernelFunction, emitted.wrapperFunction],
          }),
        };
      });
      return {
        modules,
        dependencies: [{ ecosystem: "python", name: "triton" }],
        launchWrappers: kernels.map((kernel) => ({
          hostFunctionName: stemOf(kernel.name),
          kernelName: kernel.name,
          metaParameters: kernel.launch.metaParameters ?? [],
        })),
      };
    },
  };
}
