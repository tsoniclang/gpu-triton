import type { GpuIrFunction } from "@tsonic/target-gpu";
import type { PyFunction } from "../py/model.js";
import type { ReductionPlan, TensorParameter } from "./classify.js";
import { pyName } from "./names.js";

interface ReductionEmit {
  readonly kernelFunction: PyFunction;
  readonly wrapperFunction: PyFunction;
}

// Whole-tensor sum lowered as a single-program block reduction: one arange
// over a power-of-two block covering the tensor, masked with the identity.
export function lowerReductionKernel(kernel: GpuIrFunction, plan: ReductionPlan): ReductionEmit {
  const tensors = kernel.parameters.filter((parameter): parameter is TensorParameter => parameter.kind === "tensor");
  const values = tensors.find((tensor) => tensor.name === plan.valuesTensor);
  const out = tensors.find((tensor) => tensor.name === plan.outTensor);
  const valuesName = pyName(plan.valuesTensor);
  const outName = pyName(plan.outTensor);
  const partialName = pyName(plan.partialResult);
  const valuesDimension = values?.tensor.shape[0];
  const dimName = valuesDimension !== undefined && valuesDimension.kind === "symbol" ? pyName(valuesDimension.name) : `${valuesName}_numel`;
  const kernelFunctionName = `_${pyName(kernel.name)}_kernel`;
  void out;

  return {
    kernelFunction: {
      name: kernelFunctionName,
      parameters: [`${valuesName}_ptr`, `${outName}_ptr`, dimName, "BLOCK_SIZE: tl.constexpr"],
      decorators: ["triton.jit"],
      body: [
        { kind: "assign", target: "offsets", value: "tl.arange(0, BLOCK_SIZE)" },
        { kind: "assign", target: "mask", value: `offsets < ${dimName}` },
        {
          kind: "assign",
          target: `${valuesName}_vals`,
          value: `tl.load(${valuesName}_ptr + offsets, mask=mask, other=0.0)`,
        },
        { kind: "assign", target: partialName, value: `tl.sum(${valuesName}_vals)` },
        { kind: "expression", value: `tl.store(${outName}_ptr, ${partialName})` },
      ],
    },
    wrapperFunction: {
      name: pyName(kernel.name),
      parameters: [valuesName, outName],
      decorators: [],
      body: [
        { kind: "assign", target: dimName, value: `${valuesName}.shape[0]` },
        { kind: "assign", target: "block_size", value: `triton.next_power_of_2(${dimName})` },
        { kind: "assign", target: "grid", value: "(1,)" },
        {
          kind: "expression",
          value: `${kernelFunctionName}[grid](${valuesName}, ${outName}, ${dimName}, BLOCK_SIZE=block_size)`,
        },
      ],
    },
  };
}
