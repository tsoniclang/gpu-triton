import { test } from "node:test";
import assert from "node:assert/strict";
import { printPyModule, pyName } from "../dist/index.js";

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

test("python names are sanitized deterministically", () => {
  assert.equal(pyName("%7"), "_t7");
  assert.equal(pyName("lambda"), "lambda_v");
  assert.equal(pyName("offsets"), "offsets_v");
  assert.equal(pyName("_t7"), "_t7_v");
  assert.equal(pyName("acc"), "acc");
});
