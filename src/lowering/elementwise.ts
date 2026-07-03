import type { GpuIrFunction, GpuIrOperation, GpuScalarType } from "@tsonic/target-gpu";
import { tritonElementwiseBlockSize, tritonIntrinsicRows } from "../capabilities/matrix.js";
import type { PyFunction, PyStatement } from "../py/model.js";
import type { ElementwisePlan, TensorParameter } from "./classify.js";
import { createPyNameAllocator } from "./names.js";

const binaryOperatorText: ReadonlyMap<string, string> = new Map([
  ["add", "+"],
  ["sub", "-"],
  ["mul", "*"],
  ["div", "/"],
  ["mod", "%"],
  ["lt", "<"],
  ["le", "<="],
  ["gt", ">"],
  ["ge", ">="],
  ["eq", "=="],
  ["ne", "!="],
  ["and", "&"],
  ["or", "|"],
]);

function constText(dtype: GpuScalarType, value: number | boolean): string {
  if (typeof value === "boolean") {
    return value ? "True" : "False";
  }
  if (dtype === "float32" || dtype === "float64" || dtype === "float16" || dtype === "bfloat16") {
    return Number.isInteger(value) ? `${value}.0` : `${value}`;
  }
  return `${value}`;
}

function zeroText(dtype: GpuScalarType): string {
  if (dtype === "bool") {
    return "False";
  }
  return constText(dtype, 0);
}

interface ElementwiseEmit {
  readonly kernelFunction: PyFunction;
  readonly wrapperFunction: PyFunction;
}

// SPMD-to-block lowering: the global thread index becomes a BLOCK_SIZE-wide
// offsets vector, scalar IR values become Triton vectors (scalars broadcast),
// and conditional guards become load/store masks.
export function lowerElementwiseKernel(kernel: GpuIrFunction, plan: ElementwisePlan, functionStem: string): ElementwiseEmit {
  const tensors = kernel.parameters.filter((parameter): parameter is TensorParameter => parameter.kind === "tensor");
  const scalars = kernel.parameters.filter((parameter) => parameter.kind === "scalar");

  const names = createPyNameAllocator();
  const kernelFunctionName = `_${functionStem}_kernel`;
  const wrapperName = functionStem;
  names.reserve(kernelFunctionName);
  names.reserve(wrapperName);
  for (const tensor of tensors) {
    names.nameFor(tensor.name);
  }
  for (const scalar of scalars) {
    names.nameFor(scalar.name);
  }

  const dimArgumentByTensor = new Map<string, string>();
  const dimArguments: string[] = [];
  for (const tensor of tensors) {
    const dimension = tensor.tensor.shape[0];
    if (dimension === undefined || dimension.kind !== "symbol") {
      continue;
    }
    const argument = names.nameFor(dimension.name);
    dimArgumentByTensor.set(tensor.name, argument);
    if (!dimArguments.includes(argument)) {
      dimArguments.push(argument);
    }
  }

  names.bind(plan.threadIndexResult, "offsets");

  const body: PyStatement[] = [
    { kind: "assign", target: "pid", value: "tl.program_id(0)" },
    { kind: "assign", target: "offsets", value: "pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)" },
  ];
  const emittedMasks = new Map<string, string>();

  const reference = (id: string): string => names.nameFor(id);

  const maskExpressionFor = (tensor: TensorParameter, guards: readonly string[]): string => {
    let maskName = emittedMasks.get(tensor.name);
    if (maskName === undefined) {
      maskName = names.derived(`mask_${names.nameFor(tensor.name)}`);
      const dimension = tensor.tensor.shape[0];
      const bound =
        dimension === undefined
          ? "0"
          : dimension.kind === "symbol"
            ? (dimArgumentByTensor.get(tensor.name) ?? names.nameFor(dimension.name))
            : dimension.kind === "literal"
              ? `${dimension.value}`
              : "0";
      body.push({ kind: "assign", target: maskName, value: `offsets < ${bound}` });
      emittedMasks.set(tensor.name, maskName);
    }
    return [maskName, ...guards].join(" & ");
  };

  const tensorByName = new Map(tensors.map((tensor) => [tensor.name, tensor]));

  const emitOperations = (operations: readonly GpuIrOperation[], guards: readonly string[]): void => {
    for (const operation of operations) {
      switch (operation.kind) {
        case "thread-index":
          break;
        case "const": {
          const name = names.nameFor(operation.result);
          body.push({ kind: "assign", target: name, value: constText(operation.dtype, operation.value) });
          break;
        }
        case "binary": {
          const name = names.nameFor(operation.result);
          const operator = binaryOperatorText.get(operation.operator);
          if (operator === undefined) {
            throw new Error(`Triton elementwise lowering has no operator row for '${operation.operator}' after validation accepted it.`);
          }
          body.push({
            kind: "assign",
            target: name,
            value: `(${reference(operation.left)}) ${operator} (${reference(operation.right)})`,
          });
          break;
        }
        case "unary": {
          const name = names.nameFor(operation.result);
          const text = operation.operator === "neg" ? `-(${reference(operation.operand)})` : `~(${reference(operation.operand)})`;
          body.push({ kind: "assign", target: name, value: text });
          break;
        }
        case "intrinsic": {
          const row = tritonIntrinsicRows.find((candidate) => candidate.intrinsic === operation.name);
          if (row === undefined) {
            throw new Error(`Triton elementwise lowering has no intrinsic row for '${operation.name}' after validation accepted it.`);
          }
          const name = names.nameFor(operation.result);
          const args = operation.operands.map((operand) => reference(operand)).join(", ");
          body.push({ kind: "assign", target: name, value: `${row.tritonExpression}(${args})` });
          break;
        }
        case "load": {
          const tensor = tensorByName.get(operation.tensor);
          if (tensor === undefined) {
            throw new Error(`Triton elementwise lowering references unknown tensor '${operation.tensor}' after validation accepted it.`);
          }
          const name = names.nameFor(operation.result);
          const maskExpression = maskExpressionFor(tensor, guards);
          body.push({
            kind: "assign",
            target: name,
            value: `tl.load(${names.derived(`${names.nameFor(tensor.name)}_ptr`)} + offsets, mask=${maskExpression}, other=${zeroText(operation.dtype)})`,
          });
          break;
        }
        case "store": {
          const tensor = tensorByName.get(operation.tensor);
          if (tensor === undefined) {
            throw new Error(`Triton elementwise lowering references unknown tensor '${operation.tensor}' after validation accepted it.`);
          }
          const maskExpression = maskExpressionFor(tensor, guards);
          body.push({
            kind: "expression",
            value: `tl.store(${names.derived(`${names.nameFor(tensor.name)}_ptr`)} + offsets, ${reference(operation.value)}, mask=${maskExpression})`,
          });
          break;
        }
        case "if": {
          const guard = reference(operation.condition);
          emitOperations(operation.then.operations, [...guards, guard]);
          if (operation.else !== undefined) {
            emitOperations(operation.else.operations, [...guards, `~(${guard})`]);
          }
          break;
        }
        case "return":
          break;
        default:
          throw new Error(
            `Triton elementwise lowering has no rule for operation '${(operation as { kind: string }).kind}' after classification accepted the kernel.`,
          );
      }
    }
  };
  emitOperations(kernel.body.operations, []);

  const kernelParameters = [
    ...tensors.map((tensor) => names.derived(`${names.nameFor(tensor.name)}_ptr`)),
    ...scalars.map((scalar) => names.nameFor(scalar.name)),
    ...dimArguments,
    "BLOCK_SIZE: tl.constexpr",
  ];

  const wrapperBody: PyStatement[] = [];
  const ownerOfDim = (argument: string): TensorParameter | undefined =>
    tensors.find((tensor) => dimArgumentByTensor.get(tensor.name) === argument);
  for (const argument of dimArguments) {
    const owner = ownerOfDim(argument);
    if (owner !== undefined) {
      wrapperBody.push({ kind: "assign", target: argument, value: `${names.nameFor(owner.name)}.shape[0]` });
    }
  }
  const gridDimension = kernel.launch.grid[0];
  const gridBound =
    gridDimension !== undefined && gridDimension.kind === "symbol"
      ? names.nameFor(gridDimension.name)
      : gridDimension !== undefined && gridDimension.kind === "literal"
        ? `${gridDimension.value}`
        : "1";
  wrapperBody.push({
    kind: "assign",
    target: "grid",
    value: `(triton.cdiv(${gridBound}, ${tritonElementwiseBlockSize}),)`,
  });
  wrapperBody.push({
    kind: "expression",
    value: `${kernelFunctionName}[grid](${[
      ...tensors.map((tensor) => names.nameFor(tensor.name)),
      ...scalars.map((scalar) => names.nameFor(scalar.name)),
      ...dimArguments,
      `BLOCK_SIZE=${tritonElementwiseBlockSize}`,
    ].join(", ")})`,
  });

  return {
    kernelFunction: {
      name: kernelFunctionName,
      parameters: kernelParameters,
      decorators: ["triton.jit"],
      body,
    },
    wrapperFunction: {
      name: wrapperName,
      parameters: [...tensors.map((tensor) => names.nameFor(tensor.name)), ...scalars.map((scalar) => names.nameFor(scalar.name))],
      decorators: [],
      body: wrapperBody,
    },
  };
}
