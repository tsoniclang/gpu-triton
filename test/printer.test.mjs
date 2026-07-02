import { test } from "node:test";
import assert from "node:assert/strict";
import { createPyNameAllocator, printPyModule, pyName } from "../dist/index.js";

test("printer renders imports, decorators, and nesting deterministically", () => {
  const text = printPyModule({
    imports: ["import triton"],
    functions: [
      {
        name: "f",
        parameters: ["x"],
        decorators: ["triton.jit"],
        body: [
          { kind: "assign", target: "y", value: "x + 1" },
          {
            kind: "for",
            target: "k",
            iterable: "range(0, 4)",
            body: [{ kind: "if", condition: "y > k", body: [{ kind: "expression", value: "tl.store(x, y)" }] }],
          },
          { kind: "return", value: "y" },
        ],
      },
      { name: "empty", parameters: [], decorators: [], body: [] },
    ],
  });
  assert.equal(
    text,
    `import triton


@triton.jit
def f(x):
    y = x + 1
    for k in range(0, 4):
        if y > k:
            tl.store(x, y)
    return y


def empty():
    pass
`,
  );
});

test("empty nested blocks render pass", () => {
  const text = printPyModule({
    imports: [],
    functions: [{ name: "f", parameters: [], decorators: [], body: [{ kind: "if", condition: "True", body: [] }] }],
  });
  assert.ok(text.includes("if True:\n        pass"));
});

test("module-level names are sanitized deterministically", () => {
  assert.equal(pyName("%7"), "_t7");
  assert.equal(pyName("lambda"), "lambda_v");
  assert.equal(pyName("offsets"), "offsets_v");
  assert.equal(pyName("add"), "add");
});

test("the per-kernel allocator is a bijection", () => {
  const names = createPyNameAllocator();
  assert.equal(names.nameFor("lambda"), "lambda_v");
  assert.notEqual(names.nameFor("lambda_v"), "lambda_v");
  assert.equal(names.nameFor("lambda"), "lambda_v");
  assert.equal(names.nameFor("%7"), "_t7");
  assert.notEqual(names.nameFor("_t7"), "_t7");
  assert.equal(names.nameFor("offsets"), "offsets_v");
  const allocated = ["lambda", "lambda_v", "%7", "_t7", "offsets"].map((id) => names.nameFor(id));
  assert.equal(new Set(allocated).size, allocated.length);
  const mask = names.derived("mask_out");
  assert.notEqual(names.nameFor("mask_out"), mask);
  names.reserve("taken");
  assert.notEqual(names.nameFor("taken"), "taken");
});
