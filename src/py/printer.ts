import type { PyFunction, PyModule, PyStatement } from "./model.js";

const indentUnit = "    ";

export function printPyModule(module: PyModule): string {
  const lines: string[] = [];
  for (const importLine of module.imports) {
    lines.push(importLine);
  }
  for (const fn of module.functions) {
    lines.push("");
    lines.push("");
    printFunction(fn, lines);
  }
  return `${lines.join("\n")}\n`;
}

function printFunction(fn: PyFunction, lines: string[]): void {
  for (const decorator of fn.decorators) {
    lines.push(`@${decorator}`);
  }
  lines.push(`def ${fn.name}(${fn.parameters.join(", ")}):`);
  if (fn.body.length === 0) {
    lines.push(`${indentUnit}pass`);
    return;
  }
  printStatements(fn.body, 1, lines);
}

function printStatements(statements: readonly PyStatement[], depth: number, lines: string[]): void {
  const indent = indentUnit.repeat(depth);
  for (const statement of statements) {
    switch (statement.kind) {
      case "assign":
        lines.push(`${indent}${statement.target} = ${statement.value}`);
        break;
      case "expression":
        lines.push(`${indent}${statement.value}`);
        break;
      case "for":
        lines.push(`${indent}for ${statement.target} in ${statement.iterable}:`);
        printBlock(statement.body, depth + 1, lines);
        break;
      case "if":
        lines.push(`${indent}if ${statement.condition}:`);
        printBlock(statement.body, depth + 1, lines);
        break;
      case "return":
        lines.push(statement.value === undefined ? `${indent}return` : `${indent}return ${statement.value}`);
        break;
      case "blank":
        lines.push("");
        break;
    }
  }
}

function printBlock(statements: readonly PyStatement[], depth: number, lines: string[]): void {
  if (statements.length === 0) {
    lines.push(`${indentUnit.repeat(depth)}pass`);
    return;
  }
  printStatements(statements, depth, lines);
}
