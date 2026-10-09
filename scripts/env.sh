# Source from the repository root:  source scripts/env.sh
# Puts src/ (the `attack` and `oracle` packages) on PYTHONPATH.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
