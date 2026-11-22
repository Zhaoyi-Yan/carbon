#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${ARENA_PYTHON:-python3.12}"
venv="${ARENA_VENV:-$root/.arena-venv}"

"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Choose Python 3.12 with ARENA_PYTHON"'
"$python_bin" -m venv "$venv"
"$venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12)'
"$venv/bin/python" -m pip install 'numpy==1.26.4' 'jsonschema==4.23.0'
if [[ "$(uname -s)" == Linux ]]; then
  "$venv/bin/python" -m pip install 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
else
  "$venv/bin/python" -m pip install 'torch==2.5.1'
fi
PYTHONPATH="$root/vendor" "$venv/bin/python" -c 'import numpy, torch; from zerosum_env.envs.carbon import carbon; print("Carbon referee ready", numpy.__version__, torch.__version__)'
