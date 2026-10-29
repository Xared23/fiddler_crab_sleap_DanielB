# Shared uv environment for Slurm scripts. Source from the project root.
export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PYTHONUNBUFFERED=1
mkdir -p "${UV_CACHE_DIR}"
