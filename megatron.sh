#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Megatron-LM Environment Check + Auto Fix Script
#
# Important:
#   - NO conda
#   - NO venv
#   - Use current python3
#   - unset PIP_CONSTRAINT / CONSTRAINTS_FILE
#   - use python3 -m pip install
#   - use --break-system-packages when needed
#
# Log:
#   /root/megatron_env_check_fix.log
# ============================================================

LOG="/root/megatron_env_check_fix.log"

PROJECT_BASE="${PROJECT_BASE:-/datadisk_1/projects}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_BASE}/Megatron-LM}"

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-17890}"

TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-cu124}"

echo "==== Megatron Environment Check/Fix ===="
echo "Log: ${LOG}"
echo "Project base: ${PROJECT_BASE}"
echo "Megatron dir: ${MEGATRON_DIR}"
echo

# ----------------------------
# 0. Proxy
# ----------------------------
echo "==== [0] Proxy check ===="
if ss -lnt 2>/dev/null | grep -q "${PROXY_HOST}:${PROXY_PORT}"; then
  export http_proxy="http://${PROXY_HOST}:${PROXY_PORT}"
  export https_proxy="http://${PROXY_HOST}:${PROXY_PORT}"
  export HTTP_PROXY="http://${PROXY_HOST}:${PROXY_PORT}"
  export HTTPS_PROXY="http://${PROXY_HOST}:${PROXY_PORT}"
  export all_proxy="socks5://${PROXY_HOST}:${PROXY_PORT}"
  export ALL_PROXY="socks5://${PROXY_HOST}:${PROXY_PORT}"
  echo "Proxy detected at ${PROXY_HOST}:${PROXY_PORT}"
else
  echo "No proxy detected at ${PROXY_HOST}:${PROXY_PORT}; continuing without proxy."
fi
echo

# ----------------------------
# 1. System info
# ----------------------------
echo "==== [1] System info ===="
uname -a || true
cat /etc/os-release || true
echo

if [ -f /etc/os-release ]; then
  source /etc/os-release
  OS_CODENAME="${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}"
else
  OS_CODENAME=""
fi

if [ -z "${OS_CODENAME}" ]; then
  OS_CODENAME="$(lsb_release -cs 2>/dev/null || true)"
fi

echo "Detected Ubuntu codename: ${OS_CODENAME:-unknown}"
echo

# ----------------------------
# 2. Repair apt source codename mismatch
# ----------------------------
echo "==== [2] Repair apt sources ===="

if [ -n "${OS_CODENAME}" ]; then
  TS="$(date +%F_%H%M%S)"
  mkdir -p "/root/apt_backup_${TS}"

  cp -a /etc/apt/sources.list "/root/apt_backup_${TS}/sources.list" 2>/dev/null || true
  cp -a /etc/apt/sources.list.d "/root/apt_backup_${TS}/sources.list.d" 2>/dev/null || true

  echo "Old Ubuntu codenames currently found:"
  grep -R -nE "jammy|focal|mantic|kinetic|lunar" /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null || true

  if [ "${OS_CODENAME}" = "noble" ]; then
    echo "Ubuntu 24.04 noble detected. Rewriting Ubuntu sources to noble."

    mkdir -p /etc/apt/disabled-sources

    find /etc/apt/sources.list.d -type f \( -name "*.list" -o -name "*.sources" \) \
      ! -iname "*nodesource*" \
      -exec mv {} /etc/apt/disabled-sources/ \; 2>/dev/null || true

    cat > /etc/apt/sources.list <<'EOF'
deb http://mirrors.ustc.edu.cn/ubuntu/ noble main restricted universe multiverse
deb http://mirrors.ustc.edu.cn/ubuntu/ noble-updates main restricted universe multiverse
deb http://mirrors.ustc.edu.cn/ubuntu/ noble-backports main restricted universe multiverse
deb http://mirrors.ustc.edu.cn/ubuntu/ noble-security main restricted universe multiverse
EOF

  elif [ "${OS_CODENAME}" = "jammy" ]; then
    echo "Ubuntu 22.04 jammy detected. Rewriting Ubuntu sources to jammy."

    mkdir -p /etc/apt/disabled-sources

    find /etc/apt/sources.list.d -type f \( -name "*.list" -o -name "*.sources" \) \
      ! -iname "*nodesource*" \
      -exec mv {} /etc/apt/disabled-sources/ \; 2>/dev/null || true

    cat > /etc/apt/sources.list <<'EOF'
deb http://mirrors.ustc.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.ustc.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.ustc.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
deb http://mirrors.ustc.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
EOF

  else
    echo "Detected codename ${OS_CODENAME}. Doing sed replacement only."
    sed -i -E "s/\bjammy\b/${OS_CODENAME}/g; s/\bfocal\b/${OS_CODENAME}/g; s/\bmantic\b/${OS_CODENAME}/g; s/\bkinetic\b/${OS_CODENAME}/g; s/\blunar\b/${OS_CODENAME}/g" /etc/apt/sources.list 2>/dev/null || true
    find /etc/apt/sources.list.d -type f \( -name "*.list" -o -name "*.sources" \) \
      -exec sed -i -E "s/\bjammy\b/${OS_CODENAME}/g; s/\bfocal\b/${OS_CODENAME}/g; s/\bmantic\b/${OS_CODENAME}/g; s/\bkinetic\b/${OS_CODENAME}/g; s/\blunar\b/${OS_CODENAME}/g" {} \; 2>/dev/null || true
  fi
fi

apt-get clean
rm -rf /var/lib/apt/lists/*
apt-get update

echo "Remaining old codenames after repair:"
grep -R -nE "jammy|focal|mantic|kinetic|lunar" /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null || true
echo

# ----------------------------
# 3. GPU / CUDA
# ----------------------------
echo "==== [3] GPU / CUDA info ===="
nvidia-smi || echo "nvidia-smi not found or GPU unavailable."
nvcc --version || echo "nvcc not found."
echo

# ----------------------------
# 4. APT dependencies
# ----------------------------
echo "==== [4] Install system dependencies ===="

apt-get install -y \
  git git-lfs \
  curl wget aria2 ca-certificates gnupg \
  tmux htop vim nano \
  build-essential gcc g++ make \
  cmake ninja-build pkg-config \
  python3 python3-dev python3-pip python3-venv \
  libssl-dev libffi-dev zlib1g-dev \
  libbz2-dev liblzma-dev libsqlite3-dev \
  libaio-dev \
  pciutils lsof net-tools iproute2 \
  jq

git lfs install || true
echo

# ----------------------------
# 5. Use current python3, no venv
# ----------------------------
echo "==== [5] Use current python3 environment ===="

PYTHON_BIN="$(command -v python3)"
echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "Before cleanup, pip-related variables:"
env | grep -iE "PIP|CONSTRAINT|TORCH" || true
echo

# NVIDIA containers may set constraints forcing torch==2.7.0a0+nv...
# Remove them.
unset PIP_CONSTRAINT
unset CONSTRAINTS_FILE
unset PIP_REQUIRE_VIRTUALENV
unset UV_CONSTRAINT

# Ignore global pip config if it injects constraints.
export PIP_CONFIG_FILE=/dev/null

echo "After cleanup, pip-related variables:"
env | grep -iE "PIP|CONSTRAINT|TORCH" || true
echo

# Decide whether --break-system-packages is required.
set +e
"${PYTHON_BIN}" -m pip install -U pip setuptools wheel packaging
PIP_UPGRADE_OK=$?
set -e

if [ "${PIP_UPGRADE_OK}" -ne 0 ]; then
  echo "pip upgrade failed without --break-system-packages. Retrying with it."
  PIP_EXTRA="--break-system-packages"
  "${PYTHON_BIN}" -m pip install ${PIP_EXTRA} -U pip setuptools wheel packaging
else
  PIP_EXTRA=""
fi

echo "pip extra flag: ${PIP_EXTRA:-none}"
"${PYTHON_BIN}" -m pip --version
echo

# ----------------------------
# 6. PyTorch CUDA
# ----------------------------
echo "==== [6] PyTorch CUDA check/install ===="

set +e
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
TORCH_OK=$?
set -e

if [ "${TORCH_OK}" -ne 0 ]; then
  echo "PyTorch not found. Installing PyTorch CUDA wheel: ${TORCH_CUDA_INDEX}"

  unset PIP_CONSTRAINT
  unset CONSTRAINTS_FILE
  unset PIP_REQUIRE_VIRTUALENV
  unset UV_CONSTRAINT
  export PIP_CONFIG_FILE=/dev/null

  set +e
  "${PYTHON_BIN}" -m pip install ${PIP_EXTRA:-} \
    --no-cache-dir \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}" \
    torch torchvision torchaudio
  TORCH_INSTALL_OK=$?
  set -e

  if [ "${TORCH_INSTALL_OK}" -ne 0 ] && [ "${TORCH_CUDA_INDEX}" != "cu121" ]; then
    echo "${TORCH_CUDA_INDEX} install failed. Trying cu121..."
    "${PYTHON_BIN}" -m pip install ${PIP_EXTRA:-} \
      --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio
  fi
fi

"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
assert torch.cuda.is_available(), "CUDA is not available in PyTorch."
assert torch.cuda.device_count() >= 1, "No CUDA GPU detected."
PY
echo

# ----------------------------
# 7. Python libraries
# ----------------------------
echo "==== [7] Install Megatron/data Python dependencies ===="

unset PIP_CONSTRAINT
unset CONSTRAINTS_FILE
unset PIP_REQUIRE_VIRTUALENV
unset UV_CONSTRAINT
export PIP_CONFIG_FILE=/dev/null

"${PYTHON_BIN}" -m pip install ${PIP_EXTRA:-} -U \
  numpy scipy pandas matplotlib tqdm regex requests \
  datasets transformers tokenizers sentencepiece accelerate \
  einops nltk psutil pybind11 \
  wandb tensorboard \
  zstandard jsonlines orjson xxhash ftfy \
  beautifulsoup4 lxml

"${PYTHON_BIN}" - <<'PY'
mods = [
    "numpy", "torch", "transformers", "datasets", "tokenizers",
    "sentencepiece", "wandb", "tensorboard", "zstandard",
    "jsonlines", "orjson", "xxhash", "ftfy", "bs4", "lxml"
]
for m in mods:
    try:
        __import__(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAILED -> {e}")
PY
echo

# ----------------------------
# 8. Transformer Engine
# ----------------------------
echo "==== [8] Transformer Engine install/check ===="

set +e
"${PYTHON_BIN}" - <<'PY'
import transformer_engine
print("Transformer Engine already installed:", transformer_engine.__version__)
PY
TE_OK=$?
set -e

if [ "${TE_OK}" -ne 0 ]; then
  echo "Transformer Engine missing. Trying pip install."

  unset PIP_CONSTRAINT
  unset CONSTRAINTS_FILE
  unset PIP_REQUIRE_VIRTUALENV
  unset UV_CONSTRAINT
  export PIP_CONFIG_FILE=/dev/null

  set +e
  "${PYTHON_BIN}" -m pip install ${PIP_EXTRA:-} -U "transformer-engine[pytorch]"
  TE_INSTALL_OK=$?
  set -e

  if [ "${TE_INSTALL_OK}" -ne 0 ]; then
    echo "WARNING: Transformer Engine install failed."
    echo "Megatron may still run, but fused/BF16 paths may be limited."
  fi
fi

set +e
"${PYTHON_BIN}" - <<'PY'
try:
    import transformer_engine
    print("Transformer Engine OK:", transformer_engine.__version__)
except Exception as e:
    print("Transformer Engine still unavailable:", repr(e))
PY
set -e
echo

# ----------------------------
# 9. Megatron-LM
# ----------------------------
echo "==== [9] Megatron-LM install/check ===="

mkdir -p "${PROJECT_BASE}"

if [ -d "${MEGATRON_DIR}/.git" ]; then
  echo "Megatron-LM found at ${MEGATRON_DIR}"
  cd "${MEGATRON_DIR}"
  git status --short || true
else
  echo "Cloning Megatron-LM to ${MEGATRON_DIR}"
  cd "${PROJECT_BASE}"
  git clone https://github.com/NVIDIA/Megatron-LM.git "$(basename "${MEGATRON_DIR}")"
  cd "${MEGATRON_DIR}"
fi

unset PIP_CONSTRAINT
unset CONSTRAINTS_FILE
unset PIP_REQUIRE_VIRTUALENV
unset UV_CONSTRAINT
export PIP_CONFIG_FILE=/dev/null

"${PYTHON_BIN}" -m pip install ${PIP_EXTRA:-} -e .

"${PYTHON_BIN}" - <<'PY'
try:
    import megatron
    print("Megatron import OK")
except Exception as e:
    print("Megatron import FAILED:", repr(e))
    raise
PY
echo

# ----------------------------
# 10. Node/npm + Codex optional
# ----------------------------
echo "==== [10] Node/npm/Codex optional check ===="

if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
  echo "node: $(node -v)"
  echo "npm: $(npm -v)"
else
  echo "Node/npm missing. Installing Node 20 via NodeSource."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
  echo "node: $(node -v)"
  echo "npm: $(npm -v)"
fi

if command -v codex >/dev/null 2>&1; then
  echo "codex: $(codex --version)"
else
  echo "Codex missing. Trying npm install -g @openai/codex."

  if ss -lnt 2>/dev/null | grep -q "${PROXY_HOST}:${PROXY_PORT}"; then
    npm config set proxy "http://${PROXY_HOST}:${PROXY_PORT}" || true
    npm config set https-proxy "http://${PROXY_HOST}:${PROXY_PORT}" || true
  fi

  set +e
  npm install -g @openai/codex
  CODEX_OK=$?
  set -e

  if [ "${CODEX_OK}" -ne 0 ]; then
    echo "Global npm install failed. Installing to /root/.npm-global."
    mkdir -p /root/.npm-global
    npm config set prefix /root/.npm-global
    export PATH="/root/.npm-global/bin:$PATH"
    grep -q "/root/.npm-global/bin" ~/.bashrc || echo 'export PATH="/root/.npm-global/bin:$PATH"' >> ~/.bashrc
    npm install -g @openai/codex
  fi

  codex --version || true
fi
echo

# ----------------------------
# 11. Final report
# ----------------------------
echo "==== [11] Final environment report ===="

echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip --version
echo

"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

try:
    import transformer_engine
    print("transformer_engine:", transformer_engine.__version__)
except Exception as e:
    print("transformer_engine: unavailable", repr(e))

try:
    import megatron
    print("megatron: import OK")
except Exception as e:
    print("megatron: import FAILED", repr(e))

import transformers, datasets, wandb
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("wandb:", wandb.__version__)
PY

echo
echo "==== DONE ===="
echo "Megatron path:"
echo "${MEGATRON_DIR}"

