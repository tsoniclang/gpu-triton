import type { GpuIrFunction } from "@tsonic/target-gpu";
import { tritonMatmulBlockPolicy } from "../capabilities/matrix.js";
import type { PyFunction } from "../py/model.js";
import type { MatmulPlan } from "./classify.js";
import { pyName } from "./names.js";

interface MatmulEmit {
  readonly kernelFunction: PyFunction;
  readonly wrapperFunction: PyFunction;
}

// The recognized [M,K] x [K,N] -> [M,N] accumulator pattern lowers through
// the canonical tiled Triton matmul: 2D program grid, block tiles, tl.dot
// accumulation over K blocks. Block sizes come from backend policy rows.
export function lowerMatmulKernel(kernel: GpuIrFunction, plan: MatmulPlan): MatmulEmit {
  const a = pyName(plan.a);
  const b = pyName(plan.b);
  const c = pyName(plan.c);
  const m = pyName(plan.m);
  const k = pyName(plan.k);
  const n = pyName(plan.n);
  const { blockM, blockN, blockK } = tritonMatmulBlockPolicy;
  const kernelFunctionName = `_${pyName(kernel.name)}_kernel`;

  return {
    kernelFunction: {
      name: kernelFunctionName,
      parameters: [
        `${a}_ptr`,
        `${b}_ptr`,
        `${c}_ptr`,
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
              target: `${a}_mask`,
              value: `(offs_m[:, None] < ${m}) & (k_offsets[None, :] < ${k})`,
            },
            {
              kind: "assign",
              target: `${a}_tile`,
              value: `tl.load(${a}_ptr + offs_m[:, None] * ${k} + k_offsets[None, :], mask=${a}_mask, other=0.0)`,
            },
            {
              kind: "assign",
              target: `${b}_mask`,
              value: `(k_offsets[:, None] < ${k}) & (offs_n[None, :] < ${n})`,
            },
            {
              kind: "assign",
              target: `${b}_tile`,
              value: `tl.load(${b}_ptr + k_offsets[:, None] * ${n} + offs_n[None, :], mask=${b}_mask, other=0.0)`,
            },
            { kind: "assign", target: "acc", value: `acc + tl.dot(${a}_tile, ${b}_tile)` },
          ],
        },
        { kind: "assign", target: `${c}_mask`, value: `(offs_m[:, None] < ${m}) & (offs_n[None, :] < ${n})` },
        {
          kind: "expression",
          value: `tl.store(${c}_ptr + offs_m[:, None] * ${n} + offs_n[None, :], acc, mask=${c}_mask)`,
        },
      ],
    },
    wrapperFunction: {
      name: pyName(kernel.name),
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
