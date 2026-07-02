import {
  gpuBinaryOperatorCapability,
  gpuControlIfCapability,
  gpuControlLoopCapability,
  gpuDeviceCapability,
  gpuDtypeCapability,
  gpuLaunchMetaCapability,
  gpuLayoutCapability,
  gpuMathIntrinsicCapability,
  gpuMemoryLoadCapability,
  gpuMemoryMaskedCapability,
  gpuMemoryStoreCapability,
  gpuMutableLocalCapability,
  gpuReduceCapability,
  gpuShapeSymbolicCapability,
  gpuThreadIndexCapability,
  gpuUnaryOperatorCapability,
  type GpuBackendCapabilitySet,
  type GpuMathIntrinsic,
} from "@tsonic/target-gpu";

export const tritonGpuBackendId = "triton";

// Declarative intrinsic rows: GPU math intrinsics the Triton target lowers,
// with their Triton spellings. Lowering branches on rows, never on names.
export interface TritonIntrinsicRow {
  readonly intrinsic: GpuMathIntrinsic;
  readonly tritonExpression: string;
}

export const tritonIntrinsicRows: readonly TritonIntrinsicRow[] = Object.freeze([
  { intrinsic: "sqrt", tritonExpression: "tl.sqrt" },
  { intrinsic: "exp", tritonExpression: "tl.exp" },
  { intrinsic: "log", tritonExpression: "tl.log" },
  { intrinsic: "tanh", tritonExpression: "tl.tanh" },
]);

// Backend block-size policy rows; tiling policy is Triton-specific data.
export const tritonElementwiseBlockSize = 1024;
export const tritonMatmulBlockPolicy = Object.freeze({ blockM: 64, blockN: 64, blockK: 32 });

// The initial capability matrix from the backend starter kit: float32/int32/
// bool, rank 1 elementwise and rank 2 matmul, contiguous layouts, CUDA
// devices, masked load/store, 1D/2D launch, sum reduction. Atomics, barriers,
// wide dtypes, and max reductions stay unsupported and fail closed upstream.
export const tritonCapabilities: GpuBackendCapabilitySet = Object.freeze({
  backendId: tritonGpuBackendId,
  maxTensorRank: 2,
  capabilityIds: Object.freeze([
    gpuDtypeCapability("bool"),
    gpuDtypeCapability("int32"),
    gpuDtypeCapability("float32"),
    gpuDeviceCapability("cuda"),
    gpuLayoutCapability({ kind: "contiguous" }),
    gpuShapeSymbolicCapability,
    gpuThreadIndexCapability("global"),
    gpuThreadIndexCapability("local"),
    gpuBinaryOperatorCapability("add"),
    gpuBinaryOperatorCapability("sub"),
    gpuBinaryOperatorCapability("mul"),
    gpuBinaryOperatorCapability("div"),
    gpuBinaryOperatorCapability("lt"),
    gpuBinaryOperatorCapability("le"),
    gpuBinaryOperatorCapability("gt"),
    gpuBinaryOperatorCapability("ge"),
    gpuBinaryOperatorCapability("eq"),
    gpuBinaryOperatorCapability("ne"),
    gpuBinaryOperatorCapability("and"),
    gpuBinaryOperatorCapability("or"),
    gpuUnaryOperatorCapability("neg"),
    gpuUnaryOperatorCapability("not"),
    gpuMemoryLoadCapability,
    gpuMemoryStoreCapability,
    gpuMemoryMaskedCapability,
    gpuControlIfCapability,
    gpuControlLoopCapability,
    gpuMutableLocalCapability,
    gpuLaunchMetaCapability,
    gpuReduceCapability("sum", "float32"),
    ...tritonIntrinsicRows.map((row) => gpuMathIntrinsicCapability(row.intrinsic)),
  ]),
});
