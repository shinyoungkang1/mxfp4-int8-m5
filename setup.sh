#!/bin/zsh
# Creates .venv with everything the Python scripts need (MLX requires Python >= 3.10).
set -e
cd "${0:A:h}"
if command -v uv >/dev/null; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python mlx mlx-lm numpy pyarrow matplotlib
else
  python3 -m venv .venv
  .venv/bin/pip install mlx mlx-lm numpy pyarrow matplotlib
fi
.venv/bin/python -c "import mlx.core as mx; print('mlx', mx.__version__, mx.default_device())"
