import type { GpuIrFunction } from "@tsonic/target-gpu";
import { tritonMatmulBlockPolicy } from "../capabilities/matrix.js";
import type { PyFunction } from "../py/model.js";
import type { MatmulPlan } from "./classify.js";
import { createPyNameAllocator, pyName } from "./names.js";

interface MatmulEmit {
  readonly kernelFunction: PyFunction;
  readonly wrapperFunction: PyFunction;
}

// The recognized [M,K] x [K,N] -> [M,N] accumulator pattern lowers through
// the canonical tiled Triton matmul: 2D program grid, block tiles, tl.dot
// accumulation over K blocks. Block sizes come from backend policy rows.
export function lowerMatmulKernel(kernel: GpuIrFunction, plan: MatmulPlan): MatmulEmit {
  const names = createPyNameAllocator();
  const kernelFunctionName = `_${pyName(kernel.name)}_kernel`;
  const wrapperName = pyName(kernel.name);
  names.reserve(kernelFunctionName);
  names.reserve(wrapperName);
  const a = names.nameFor(plan.a);
  const b = names.nameFor(plan.b);
  const c = names.nameFor(plan.c);
  const m = names.nameFor(plan.m);
  const k = names.nameFor(plan.k);
  const n = names.nameFor(plan.n);
  const aPtr = names.derived(`${a}_ptr`);
  const bPtr = names.derived(`${b}_ptr`);
  const cPtr = names.derived(`${c}_ptr`);
  const aMask = names.derived(`${a}_mask`);
  const bMask = names.derived(`${b}_mask`);
  const cMask = names.derived(`${c}_mask`);
  const aTile = names.derived(`${a}_tile`);
  const bTile = names.derived(`${b}_tile`);
  const { blockM, blockN, blockK } = tritonMatmulBlockPolicy;

  return {
    kernelFunction: {
      name: kernelFunctionName,
      parameters: [
        aPtr,
        bPtr,
        cPtr,
        m,
        n,
        k,
        "BLOCK_M: tl.constexpr",
        "BLOCK_N: tl.constexpr",
        "BLOCK_K: tl.constexpr",
      ],
      decorators: ["triton.jit"],
      body: [
        { kind: "assign", target: "pid_m", value: "tl.program_id(0)" },
        { kind: "assign", target: "pid_n", value: "tl.program_id(1)" },
        { kind: "assign", target: "offs_m", value: "pid_m * BLOCK_M + tl.arange(0, BLOCK_M)" },
        { kind: "assign", target: "offs_n", value: "pid_n * BLOCK_N + tl.arange(0, BLOCK_N)" },
        { kind: "assign", target: "offs_k", value: "tl.arange(0, BLOCK_K)" },
        { kind: "assign", target: "acc", value: "tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)" },
        {
          kind: "for",
          target: "k_block",
          iterable: `range(0, tl.cdiv(${k}, BLOCK_K))`,
          body: [
            { kind: "assign", target: "k_offsets", value: "k_block * BLOCK_K + offs_k" },
            {
              kind: "assign",
              target: aMask,
              value: `(offs_m[:, None] < ${m}) & (k_offsets[None, :] < ${k})`,
            },
            {
              kind: "assign",
              target: aTile,
              value: `tl.load(${aPtr} + offs_m[:, None] * ${k} + k_offsets[None, :], mask=${aMask}, other=0.0)`,
            },
            {
              kind: "assign",
              target: bMask,
              value: `(k_offsets[:, None] < ${k}) & (offs_n[None, :] < ${n})`,
            },
            {
              kind: "assign",
              target: bTile,
              value: `tl.load(${bPtr} + k_offsets[:, None] * ${n} + offs_n[None, :], mask=${bMask}, other=0.0)`,
            },
            { kind: "assign", target: "acc", value: `acc + tl.dot(${aTile}, ${bTile})` },
          ],
        },
        { kind: "assign", target: cMask, value: `(offs_m[:, None] < ${m}) & (offs_n[None, :] < ${n})` },
        {
          kind: "expression",
          value: `tl.store(${cPtr} + offs_m[:, None] * ${n} + offs_n[None, :], acc, mask=${cMask})`,
        },
      ],
    },
    wrapperFunction: {
      name: wrapperName,
      parameters: [a, b, c],
      decorators: [],
      body: [
        { kind: "assign", target: m, value: `${a}.shape[0]` },
        { kind: "assign", target: k, value: `${a}.shape[1]` },
        { kind: "assign", target: n, value: `${b}.shape[1]` },
        { kind: "assign", target: "grid", value: `(triton.cdiv(${m}, ${blockM}), triton.cdiv(${n}, ${blockN}))` },
        {
          kind: "expression",
          value: `${kernelFunctionName}[grid](${a}, ${b}, ${c}, ${m}, ${n}, ${k}, BLOCK_M=${blockM}, BLOCK_N=${blockN}, BLOCK_K=${blockK})`,
        },
      ],
    },
  };
}
