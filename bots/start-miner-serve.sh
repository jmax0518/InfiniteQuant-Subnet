#!/usr/bin/env bash
# Local REST intake for dashboard Submit → on-chain SN89 commitment.
# Uses the working miner install + GOLD/sn89 wallet.
set -euo pipefail

MINER_ROOT="${SN89_MINER_ROOT:-/root/MVTRX_08_05/InfiniteQuant-Subnet}"
WALLET_NAME="${WALLET_NAME:-GOLD}"
WALLET_HOTKEY="${WALLET_HOTKEY:-iq89}"
HOST="${SN89_SERVE_HOST:-127.0.0.1}"
PORT="${SN89_SERVE_PORT:-8089}"

if [[ -f /root/.sn89/env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/.sn89/env
  set +a
fi

cd "$MINER_ROOT"
export PYTHONUNBUFFERED=1
exec "${SN89_MINER_PYTHON:-$MINER_ROOT/.venv/bin/python}" neurons/miner.py \
  --wallet.name "$WALLET_NAME" \
  --wallet.hotkey "$WALLET_HOTKEY" \
  serve --host "$HOST" --port "$PORT"
