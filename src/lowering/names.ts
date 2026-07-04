// Deterministic, collision-safe Python name allocation. Each kernel gets one
// allocator covering its jit function and wrapper: distinct IR ids can never
// map to the same Python identifier, because every allocated name joins one
// used-set and candidates are suffixed until free. Extraction temporaries
// (%N) become _tN. Sanitization exists only for Python validity and path
// safety: source-visible names are never recased, and ids that are already
// valid identifiers pass through byte-for-byte unless they collide.

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
  // Names the emitters themselves assign.
  "acc",
  "block_size",
  "grid",
  "k_block",
  "k_offsets",
  "mask",
  "offs_k",
  "offs_m",
  "offs_n",
  "offsets",
  "pid",
  "pid_m",
  "pid_n",
  "tl",
  "triton",
]);

export interface PyNameAllocator {
  // Bijective id-to-name mapping: the same id always returns the same name,
  // and no two ids share one.
  nameFor(id: string): string;
  // Unique emitter-derived names (pointer args, masks, tiles) memoized per key.
  derived(key: string): string;
  // Marks a name the emitter uses verbatim (function names) as taken.
  reserve(name: string): void;
  // Pins an id to an already-reserved emitter name (the thread index vector).
  bind(id: string, name: string): void;
}

// Every allocated name is a valid Python identifier: invalid characters map
// to underscores, empty results become `value`, and leading digits gain an
// `n` prefix. Sanitization happens before the used-set claim, so ids that
// sanitize to the same identifier (a-b and a_b) still allocate distinct names.
export function sanitizePyIdentifier(text: string): string {
  const replaced = text.replace(/[^A-Za-z0-9_]/gu, "_");
  const nonEmpty = replaced.length === 0 ? "value" : replaced;
  return /^[0-9]/u.test(nonEmpty) ? `n${nonEmpty}` : nonEmpty;
}

export function createPyNameAllocator(): PyNameAllocator {
  const assigned = new Map<string, string>();
  const derivedNames = new Map<string, string>();
  const used = new Set(reservedPythonNames);
  const claim = (base: string): string => {
    let candidate = sanitizePyIdentifier(base);
    while (used.has(candidate)) {
      candidate = `${candidate}_v`;
    }
    used.add(candidate);
    return candidate;
  };
  return {
    nameFor(id) {
      const existing = assigned.get(id);
      if (existing !== undefined) {
        return existing;
      }
      const name = claim(id.startsWith("%") ? `_t${id.slice(1)}` : id);
      assigned.set(id, name);
      return name;
    },
    derived(key) {
      const existing = derivedNames.get(key);
      if (existing !== undefined) {
        return existing;
      }
      const name = claim(key);
      derivedNames.set(key, name);
      return name;
    },
    reserve(name) {
      used.add(name);
    },
    bind(id, name) {
      assigned.set(id, name);
    },
  };
}

// Single-name sanitization for contexts with no sibling names to collide
// with. Anything that shares a namespace must go through an allocator.
export function pyName(id: string): string {
  const base = sanitizePyIdentifier(id.startsWith("%") ? `_t${id.slice(1)}` : id);
  return reservedPythonNames.has(base) ? `${base}_v` : base;
}
