import type { GpuIrFunction } from "@tsonic/target-gpu";
import type { PyFunction } from "../py/model.js";
import type { ReductionPlan, TensorParameter } from "./classify.js";
import { createPyNameAllocator } from "./names.js";

interface ReductionEmit {
  readonly kernelFunction: PyFunction;
  readonly wrapperFunction: PyFunction;
}

// Whole-tensor sum lowered as a single-program block reduction: one arange
// over a power-of-two block covering the tensor, masked with the identity.
export function lowerReductionKernel(kernel: GpuIrFunction, plan: ReductionPlan, functionStem: string): ReductionEmit {
  const tensors = kernel.parameters.filter((parameter): parameter is TensorParameter => parameter.kind === "tensor");
  const values = tensors.find((tensor) => tensor.name === plan.valuesTensor);
  const out = tensors.find((tensor) => tensor.name === plan.outTensor);
  const names = createPyNameAllocator();
  const kernelFunctionName = `_${functionStem}_kernel`;
  const wrapperName = functionStem;
  names.reserve(kernelFunctionName);
  names.reserve(wrapperName);
  const valuesName = names.nameFor(plan.valuesTensor);
  const outName = names.nameFor(plan.outTensor);
  const partialName = names.nameFor(plan.partialResult);
  const valuesDimension = values?.tensor.shape[0];
  const dimName =
    valuesDimension !== undefined && valuesDimension.kind === "symbol"
      ? names.nameFor(valuesDimension.name)
      : names.derived(`${valuesName}_numel`);
  const valuesPtr = names.derived(`${valuesName}_ptr`);
  const outPtr = names.derived(`${outName}_ptr`);
  const valuesVals = names.derived(`${valuesName}_vals`);
  void out;

  return {
    kernelFunction: {
      name: kernelFunctionName,
      parameters: [valuesPtr, outPtr, dimName, "BLOCK_SIZE: tl.constexpr"],
      decorators: ["triton.jit"],
      body: [
        { kind: "assign", target: "offsets", value: "tl.arange(0, BLOCK_SIZE)" },
        { kind: "assign", target: "mask", value: `offsets < ${dimName}` },
        {
          kind: "assign",
          target: valuesVals,
          value: `tl.load(${valuesPtr} + offsets, mask=mask, other=0.0)`,
        },
        { kind: "assign", target: partialName, value: `tl.sum(${valuesVals})` },
        { kind: "expression", value: `tl.store(${outPtr}, ${partialName})` },
      ],
    },
    wrapperFunction: {
      name: wrapperName,
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
