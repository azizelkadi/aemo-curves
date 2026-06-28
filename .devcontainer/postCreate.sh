#!/usr/bin/env bash
# Runs once after the container is created. Builds the project environment in
# the /opt/venv volume from the lockfile, then does a quick sanity check.
set -euo pipefail

# A freshly created named volume mounts root-owned; claim it for the dev user.
sudo chown -R "$(id -u):$(id -g)" /opt/venv

# Install runtime + dev deps exactly as locked (CPU-only torch on Linux).
uv sync --extra dev --frozen

echo "--- environment ready ---"
echo "claude:  $(command -v claude || echo 'NOT FOUND')"
uv run python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
uv run aemo-run --help >/dev/null && echo "aemo-run: OK"
