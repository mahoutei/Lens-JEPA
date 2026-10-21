#!/usr/bin/env bash
# Install the DLens agent dependencies.
#
# DLens declares llama-cpp-python, numba, outlines, mcp, pandas and xmltodict,
# none of which the planner/report path uses: dlens/agents/_models.py imports
# llama_cpp lazily. llama-cpp-python additionally needs cmake and a C++
# compiler at install time. So DLens is installed from a pinned commit with
# --no-deps, and only the three direct dependencies are added.
#
#   bash scripts/install_agent_deps.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DLENS_SHA="${DLENS_SHA:-d0d6e4937932e29a1df208178e3d386cdc84ebb2}"
# The repository's own virtual environment when it exists, otherwise the
# interpreter the environment provides (CI has no .venv).
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi
UV="${UV:-uv}"

"$PY" -c 'import sys; sys.exit("the agent extra needs Python >= 3.12, got " + sys.version.split()[0]) if sys.version_info < (3, 12) else None'

# uv is the fast path; the CI runners have no uv, so fall back to pip.
if command -v "$UV" >/dev/null 2>&1; then
    install() { "$UV" pip install --python "$PY" "$@"; }
else
    install() { "$PY" -m pip install "$@"; }
fi

install --no-deps "git+https://github.com/ML4SCI/DeepLense-AI-Scientist.git@${DLENS_SHA}"
install "pydantic-ai>=2.0" "loguru>=0.7" "python-dotenv>=1.0"

"$PY" -c "import dlens; print('dlens ok')"
"$PY" -c "from dlens.agents._experiment_planner import ExperimentPlanner; print('planner ok')"
