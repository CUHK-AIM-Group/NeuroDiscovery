#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_REPO_ROOT="$(cd "$DESKTOP_DIR/.." && pwd)"

CONDA_EXE="${CONDA_EXE:-}"
CONDA_ENV="${CONDA_ENV:-neuroclaw}"
PYTHON_VERSION="${PYTHON_VERSION:-}"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$DESKTOP_DIR/runtime}"
REQUIREMENTS="${REQUIREMENTS:-$DESKTOP_DIR/runtime-requirements.txt}"
SKIP_PYTHON=0
SKIP_BACKEND=0

usage() {
  cat <<'USAGE'
Usage: prepare-bundled-runtime-mac.sh [options]

Options:
  --conda-exe PATH       Conda executable. Defaults to miniforge/miniconda/anaconda under $HOME.
  --conda-env NAME       Source conda environment used to choose the Python minor version. Default: neuroclaw.
  --python-version VER   Python minor version for the bundled runtime, for example 3.11.
  --repo-root PATH       NeuroClaw repository root. Default: parent of desktop/.
  --runtime-root PATH    Output runtime directory. Default: desktop/runtime.
  --requirements PATH    Runtime requirements file. Default: desktop/runtime-requirements.txt.
  --skip-python          Reuse an existing runtime/python directory.
  --skip-backend         Reuse an existing runtime/backend directory.
  -h, --help             Show this help.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --conda-exe)
      CONDA_EXE="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --runtime-root)
      RUNTIME_ROOT="$2"
      shift 2
      ;;
    --requirements)
      REQUIREMENTS="$2"
      shift 2
      ;;
    --skip-python)
      SKIP_PYTHON=1
      shift
      ;;
    --skip-backend)
      SKIP_BACKEND=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

step() {
  printf '\n==> %s\n' "$1"
}

assert_file() {
  if [ ! -f "$1" ]; then
    echo "$2" >&2
    exit 1
  fi
}

assert_dir() {
  if [ ! -d "$1" ]; then
    echo "$2" >&2
    exit 1
  fi
}

find_default_conda() {
  for candidate in \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda"
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

conda_env_exists() {
  "$CONDA_EXE" env list | awk '{print $1}' | grep -Fxq "$1"
}

detect_python_version() {
  if [ -n "$PYTHON_VERSION" ]; then
    printf '%s\n' "$PYTHON_VERSION"
    return 0
  fi

  if conda_env_exists "$CONDA_ENV"; then
    "$CONDA_EXE" run -n "$CONDA_ENV" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    return 0
  fi

  printf "Conda env '%s' was not found; using base Python minor version instead.\n" "$CONDA_ENV" >&2
  "$CONDA_EXE" run -n base python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

copy_root_file_if_exists() {
  local name="$1"
  if [ -f "$REPO_ROOT/$name" ]; then
    cp "$REPO_ROOT/$name" "$BACKEND_TARGET/$name"
  fi
}

copy_optimized_logo_if_exists() {
  local source="$REPO_ROOT/materials/logo.png"
  local target_dir="$BACKEND_TARGET/materials"
  local target="$target_dir/logo.png"
  if [ ! -f "$source" ]; then
    return 0
  fi

  mkdir -p "$target_dir"
  if command -v sips >/dev/null 2>&1; then
    sips -Z 512 "$source" --out "$target" >/dev/null
  else
    cp "$source" "$target"
  fi
}

stage_backend_dir() {
  local dir_name="$1"
  local source="$REPO_ROOT/$dir_name"
  local target="$BACKEND_TARGET/$dir_name"
  if [ ! -d "$source" ]; then
    return 0
  fi

  mkdir -p "$target"
  rsync -a --delete \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='__pycache__' \
    --exclude='dist' \
    --exclude='build' \
    --exclude='node_modules' \
    --exclude='data' \
    --exclude='models' \
    --exclude='papers' \
    --exclude='runs' \
    --exclude='logs' \
    --exclude='output' \
    --exclude='materials' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.log' \
    --exclude='.env' \
    "$source/" "$target/"
}

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
mkdir -p "$RUNTIME_ROOT"
RUNTIME_ROOT="$(cd "$RUNTIME_ROOT" && pwd)"
REQUIREMENTS="$(cd "$(dirname "$REQUIREMENTS")" && pwd)/$(basename "$REQUIREMENTS")"
PYTHON_TARGET="$RUNTIME_ROOT/python"
BACKEND_TARGET="$RUNTIME_ROOT/backend"
PYTHON_EXE="$PYTHON_TARGET/bin/python"
CONDA_UNPACK_EXE="$PYTHON_TARGET/bin/conda-unpack"

assert_dir "$REPO_ROOT" "Repo root not found: $REPO_ROOT"

if [ "$SKIP_PYTHON" -eq 0 ]; then
  if [ -z "$CONDA_EXE" ]; then
    CONDA_EXE="$(find_default_conda || true)"
  fi
  assert_file "$CONDA_EXE" "Conda executable not found. Pass --conda-exe PATH."
  assert_file "$REQUIREMENTS" "Runtime requirements file not found: $REQUIREMENTS"

  step "Create relocatable bundled Python prefix from conda env '$CONDA_ENV'"
  PYTHON_MINOR="$(detect_python_version)"
  rm -rf "$PYTHON_TARGET"
  "$CONDA_EXE" create -y -p "$PYTHON_TARGET" --copy "python=$PYTHON_MINOR" pip
  assert_file "$PYTHON_EXE" "Bundled Python executable was not created: $PYTHON_EXE"

  step "Install desktop runtime dependencies"
  "$PYTHON_EXE" -m pip install -r "$REQUIREMENTS"
  "$PYTHON_EXE" -m pip install conda-pack
  assert_file "$PYTHON_TARGET/bin/conda-pack" "conda-pack was not created: $PYTHON_TARGET/bin/conda-pack"

  step "Make bundled Python prefix relocatable"
  PACKED_PYTHON="$RUNTIME_ROOT/python-packed.tar.gz"
  "$PYTHON_TARGET/bin/conda-pack" -p "$PYTHON_TARGET" -o "$PACKED_PYTHON" --force
  rm -rf "$PYTHON_TARGET"
  mkdir -p "$PYTHON_TARGET"
  tar -xzf "$PACKED_PYTHON" -C "$PYTHON_TARGET"
  rm -f "$PACKED_PYTHON"
  assert_file "$CONDA_UNPACK_EXE" "conda-unpack was not created: $CONDA_UNPACK_EXE"

  step "Trim Python bytecode caches"
  find "$PYTHON_TARGET" -type d -name '__pycache__' -prune -exec rm -rf {} +
  find "$PYTHON_TARGET" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
fi

if [ "$SKIP_BACKEND" -eq 0 ]; then
  step "Stage NeuroClaw backend source"
  rm -rf "$BACKEND_TARGET"
  mkdir -p "$BACKEND_TARGET"

  stage_backend_dir "core"
  stage_backend_dir "skills"
  stage_backend_dir "neurooracle"

  copy_root_file_if_exists "LICENSE"
  copy_root_file_if_exists "README.md"
  copy_root_file_if_exists "README_zh.md"
  copy_root_file_if_exists "SOUL.md"
  copy_root_file_if_exists "pyproject.toml"
  copy_optimized_logo_if_exists

  cat > "$BACKEND_TARGET/neuroclaw_environment.json" <<'JSON'
{
  "setup_type": "bundled",
  "python_path": "bundled",
  "conda_env": "",
  "llm_backend": {
    "provider": "openai",
    "model": "gpt-5.5",
    "base_url": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY",
    "available_models": [
      {
        "provider": "openai",
        "model": "gpt-5.5",
        "label": "OpenAI / gpt-5.5"
      }
    ]
  },
  "cuda": {
    "device": "cpu"
  },
  "toolchain": {},
  "compression_mode": "stub"
}
JSON
fi

cat > "$RUNTIME_ROOT/runtime-manifest.json" <<JSON
{
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "conda_env": "$(json_escape "$CONDA_ENV")",
  "repo_root": "$(json_escape "$REPO_ROOT")",
  "runtime_root": "$(json_escape "$RUNTIME_ROOT")",
  "python_target": "$(json_escape "$PYTHON_TARGET")",
  "backend_target": "$(json_escape "$BACKEND_TARGET")",
  "requirements": "$(json_escape "$REQUIREMENTS")",
  "platform": "$(json_escape "$(uname -s)-$(uname -m)")",
  "strategy": "conda-prefix-relocatable"
}
JSON

printf '\nBundled macOS runtime staged at %s\n' "$RUNTIME_ROOT"
printf 'Next: npm run dist:mac\n'
