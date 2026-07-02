#!/usr/bin/env bash
set -euo pipefail

# The tsonic and tsonic-gpu repositories are read-only from Triton backend
# work. This build never writes into them: it requires their prebuilt outputs
# and points type resolution at the existing dist declarations.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TSONIC_ROOT="$(cd "$REPO_ROOT/../tsonic" && pwd -P)"
TARGET_GPU_ROOT="$(cd "$REPO_ROOT/../tsonic-gpu" && pwd -P)"

required_dist_outputs=(
  "$TSONIC_ROOT/packages/target-api/dist/index.d.ts"
  "$TSONIC_ROOT/packages/tsts/dist/src/index.d.ts"
  "$TARGET_GPU_ROOT/dist/index.d.ts"
)

for output in "${required_dist_outputs[@]}"; do
  if [[ ! -f "$output" ]]; then
    echo "FAIL: missing prebuilt output $output" >&2
    echo "Build the tsonic and tsonic-gpu packages first (they are not built from gpu-triton)." >&2
    exit 1
  fi
done

mkdir -p "$REPO_ROOT/.temp/build"
CANONICAL_TSCONFIG="$REPO_ROOT/.temp/build/tsconfig.canonical-tsonic.json"
cat > "$CANONICAL_TSCONFIG" <<EOF
{
  "extends": "../../tsconfig.json",
  "compilerOptions": {
    "paths": {
      "@tsonic/tsts": ["$TSONIC_ROOT/packages/tsts/dist/src/index.d.ts"],
      "@tsonic/target-api": ["$TSONIC_ROOT/packages/target-api/dist/index.d.ts"],
      "@tsonic/target-gpu": ["$TARGET_GPU_ROOT/dist/index.d.ts"]
    }
  }
}
EOF

"$TSONIC_ROOT/scripts/build/tsgo-project.sh" "$CANONICAL_TSCONFIG" --pretty false
