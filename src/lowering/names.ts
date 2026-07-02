// Deterministic Python name selection for IR value ids, parameters, and
// shape symbols. Extraction temporaries (%N) become _tN; anything colliding
// with Python keywords or names this backend emits gets a _v suffix.

const reservedPythonNames = new Set([
  "False",
  "None",
  "True",
  "and",
  "as",
  "assert",
  "async",
  "await",
  "break",
  "case",
  "class",
  "continue",
  "def",
  "del",
  "elif",
  "else",
  "except",
  "finally",
  "for",
  "from",
  "global",
  "if",
  "import",
  "in",
  "is",
  "lambda",
  "match",
  "nonlocal",
  "not",
  "or",
  "pass",
  "raise",
  "return",
  "try",
  "while",
  "with",
  "yield",
  "grid",
  "mask",
  "offsets",
  "pid",
  "tl",
  "triton",
]);

export function pyName(id: string): string {
  const base = id.startsWith("%") ? `_t${id.slice(1)}` : id;
  if (reservedPythonNames.has(base) || /^_t\d+$/u.test(id)) {
    return `${base}_v`;
  }
  return base;
}
