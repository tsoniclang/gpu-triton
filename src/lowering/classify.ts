import type { GpuIrFunction, GpuIrOperation, GpuKernelParameter } from "@tsonic/target-gpu";

// Every kernel must classify into one of the supported Triton lowering
// paths before any code is produced. Anything else is a deterministic
// backend diagnostic; there is no partial or substitute lowering.

export type TensorParameter = Extract<GpuKernelParameter, { kind: "tensor" }>;

export interface ElementwisePlan {
  readonly kind: "elementwise";
  readonly threadIndexResult: string;
}

export interface ReductionPlan {
  readonly kind: "reduction";
  readonly valuesTensor: string;
  readonly outTensor: string;
  readonly partialResult: string;
}

export interface MatmulPlan {
  readonly kind: "matmul";
  readonly a: string;
  readonly b: string;
  readonly c: string;
  readonly m: string;
  readonly k: string;
  readonly n: string;
  readonly rowResult: string;
  readonly colResult: string;
}

export type TritonKernelPlan = ElementwisePlan | ReductionPlan | MatmulPlan;

export interface ClassificationFailure {
  readonly capabilityId: string;
  readonly message: string;
}

export type ClassificationResult =
  | { readonly plan: TritonKernelPlan }
  | { readonly failure: ClassificationFailure };

export function classifyTritonKernel(kernel: GpuIrFunction): ClassificationResult {
  const matmul = classifyMatmul(kernel);
  if (matmul !== undefined) {
    return { plan: matmul };
  }
  const reduction = classifyReduction(kernel);
  if (reduction !== undefined) {
    return { plan: reduction };
  }
  return classifyElementwise(kernel);
}

function tensorParameters(kernel: GpuIrFunction): readonly TensorParameter[] {
  return kernel.parameters.filter((parameter): parameter is TensorParameter => parameter.kind === "tensor");
}

function shapeSymbol(parameter: TensorParameter, dimension: number): string | undefined {
  const entry = parameter.tensor.shape[dimension];
  return entry !== undefined && entry.kind === "symbol" ? entry.name : undefined;
}

function operationsWithoutReturn(operations: readonly GpuIrOperation[]): readonly GpuIrOperation[] {
  return operations.filter((operation) => operation.kind !== "return");
}

// [M,K] x [K,N] -> [M,N] with a loop-carried accumulator; lowered through
// the dedicated tiled path (tl.dot with backend block policy).
function classifyMatmul(kernel: GpuIrFunction): MatmulPlan | undefined {
  const tensors = tensorParameters(kernel);
  if (tensors.length !== 3 || kernel.parameters.length !== 3) {
    return undefined;
  }
  const [a, b, c] = tensors;
  if (a === undefined || b === undefined || c === undefined) {
    return undefined;
  }
  if (tensors.some((tensor) => tensor.tensor.rank !== 2 || tensor.tensor.elementType !== "float32")) {
    return undefined;
  }
  const m = shapeSymbol(a, 0);
  const k = shapeSymbol(a, 1);
  const n = shapeSymbol(b, 1);
  if (m === undefined || k === undefined || n === undefined) {
    return undefined;
  }
  if (shapeSymbol(b, 0) !== k || shapeSymbol(c, 0) !== m || shapeSymbol(c, 1) !== n) {
    return undefined;
  }

  const operations = operationsWithoutReturn(kernel.body.operations);
  if (operations.length !== 7) {
    return undefined;
  }
  const [rowOp, colOp, zeroOp, localOp, lowerOp, loopOp, storeOp] = operations;
  if (
    rowOp?.kind !== "thread-index" ||
    rowOp.space !== "global" ||
    rowOp.dimension !== 0 ||
    colOp?.kind !== "thread-index" ||
    colOp.space !== "global" ||
    colOp.dimension !== 1 ||
    zeroOp?.kind !== "const" ||
    localOp?.kind !== "local" ||
    localOp.initial !== zeroOp.result ||
    lowerOp?.kind !== "const" ||
    lowerOp.value !== 0 ||
    loopOp?.kind !== "loop" ||
    loopOp.lowerBound !== lowerOp.result ||
    loopOp.upperBound !== k ||
    storeOp?.kind !== "store" ||
    storeOp.tensor !== c.name ||
    storeOp.value !== localOp.result ||
    storeOp.indices.length !== 2 ||
    storeOp.indices[0] !== rowOp.result ||
    storeOp.indices[1] !== colOp.result
  ) {
    return undefined;
  }

  const body = operationsWithoutReturn(loopOp.body.operations);
  if (body.length !== 5) {
    return undefined;
  }
  const [loadA, loadB, mulOp, addOp, assignOp] = body;
  if (
    loadA?.kind !== "load" ||
    loadA.tensor !== a.name ||
    loadA.indices[0] !== rowOp.result ||
    loadA.indices[1] !== loopOp.counter ||
    loadB?.kind !== "load" ||
    loadB.tensor !== b.name ||
    loadB.indices[0] !== loopOp.counter ||
    loadB.indices[1] !== colOp.result ||
    mulOp?.kind !== "binary" ||
    mulOp.operator !== "mul" ||
    addOp?.kind !== "binary" ||
    addOp.operator !== "add" ||
    assignOp?.kind !== "assign" ||
    assignOp.target !== localOp.result
  ) {
    return undefined;
  }

  return {
    kind: "matmul",
    a: a.name,
    b: b.name,
    c: c.name,
    m,
    k,
    n,
    rowResult: rowOp.result,
    colResult: colOp.result,
  };
}

// Whole-tensor block reduction written to out[0] by lane 0.
function classifyReduction(kernel: GpuIrFunction): ReductionPlan | undefined {
  const tensors = tensorParameters(kernel);
  if (tensors.length !== 2 || kernel.parameters.length !== 2) {
    return undefined;
  }
  const [values, out] = tensors;
  if (values === undefined || out === undefined) {
    return undefined;
  }
  if (values.tensor.rank !== 1 || out.tensor.rank !== 1 || values.tensor.elementType !== "float32") {
    return undefined;
  }
  const operations = operationsWithoutReturn(kernel.body.operations);
  if (operations.length !== 5) {
    return undefined;
  }
  const [reduceOp, laneOp, zeroOp, guardOp, ifOp] = operations;
  if (
    reduceOp?.kind !== "reduce" ||
    reduceOp.operator !== "sum" ||
    reduceOp.operand !== values.name ||
    laneOp?.kind !== "thread-index" ||
    laneOp.space !== "local" ||
    zeroOp?.kind !== "const" ||
    zeroOp.value !== 0 ||
    guardOp?.kind !== "binary" ||
    guardOp.operator !== "eq" ||
    ifOp?.kind !== "if" ||
    ifOp.condition !== guardOp.result ||
    ifOp.else !== undefined
  ) {
    return undefined;
  }
  const body = operationsWithoutReturn(ifOp.then.operations);
  const [indexOp, storeOp] = body;
  if (
    body.length !== 2 ||
    indexOp?.kind !== "const" ||
    indexOp.value !== 0 ||
    storeOp?.kind !== "store" ||
    storeOp.tensor !== out.name ||
    storeOp.value !== reduceOp.result
  ) {
    return undefined;
  }
  return { kind: "reduction", valuesTensor: values.name, outTensor: out.name, partialResult: reduceOp.result };
}

// Rank-1 elementwise kernels over one global thread index: every load and
// store is indexed exactly by that index, guards become masks.
function classifyElementwise(kernel: GpuIrFunction): ClassificationResult {
  const tensors = tensorParameters(kernel);
  if (tensors.some((tensor) => tensor.tensor.rank !== 1)) {
    return {
      failure: {
        capabilityId: "triton.lowering.kernel-shape",
        message: `Kernel '${kernel.name}' does not match a Triton lowering path: rank-2 tensors lower only through the tiled matmul pattern.`,
      },
    };
  }
  if ((kernel.launch.metaParameters ?? []).length > 0) {
    return {
      failure: {
        capabilityId: "triton.lowering.launch-meta",
        message: `Kernel '${kernel.name}' declares launch meta parameters, which the Triton elementwise path does not consume; meta-driven kernels have no Triton lowering.`,
      },
    };
  }
  let threadIndexResult: string | undefined;
  const failure = walkElementwise(kernel, kernel.body.operations, {
    seeThreadIndex: (result) => {
      if (threadIndexResult !== undefined) {
        return "multiple thread indices";
      }
      threadIndexResult = result;
      return undefined;
    },
    threadIndex: () => threadIndexResult,
  });
  if (failure !== undefined) {
    return {
      failure: {
        capabilityId: "triton.lowering.kernel-shape",
        message: `Kernel '${kernel.name}' does not match a Triton lowering path: ${failure}.`,
      },
    };
  }
  if (threadIndexResult === undefined) {
    return {
      failure: {
        capabilityId: "triton.lowering.kernel-shape",
        message: `Kernel '${kernel.name}' does not match a Triton lowering path: elementwise kernels index by one global thread index.`,
      },
    };
  }
  return { plan: { kind: "elementwise", threadIndexResult } };
}

interface ElementwiseWalk {
  seeThreadIndex(result: string): string | undefined;
  threadIndex(): string | undefined;
}

function walkElementwise(
  kernel: GpuIrFunction,
  operations: readonly GpuIrOperation[],
  walk: ElementwiseWalk,
): string | undefined {
  for (const operation of operations) {
    switch (operation.kind) {
      case "thread-index": {
        if (operation.space !== "global" || operation.dimension !== 0) {
          return "elementwise kernels use only the global thread index of dimension 0";
        }
        const problem = walk.seeThreadIndex(operation.result);
        if (problem !== undefined) {
          return problem;
        }
        break;
      }
      case "load":
      case "store": {
        const index = operation.indices[0];
        if (operation.indices.length !== 1 || index !== walk.threadIndex()) {
          return "loads and stores must index tensors exactly by the global thread index";
        }
        break;
      }
      case "if": {
        const problem = walkElementwise(kernel, operation.then.operations, walk);
        if (problem !== undefined) {
          return problem;
        }
        if (operation.else !== undefined) {
          const elseProblem = walkElementwise(kernel, operation.else.operations, walk);
          if (elseProblem !== undefined) {
            return elseProblem;
          }
        }
        break;
      }
      case "const":
      case "binary":
      case "unary":
      case "intrinsic":
      case "return":
        break;
      default:
        return `operation '${operation.kind}' has no Triton elementwise lowering`;
    }
  }
  return undefined;
}
