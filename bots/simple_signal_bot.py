#!/usr/bin/env python3
"""Minimal SN89 signal bot (TEMPLATE — not financial advice).

Pulls free Binance USDT candles, applies a tiny SMA crossover, and POSTs
pair + direction to your local miner `serve` intake.

Prereqs (two terminals):
  1) python neurons/miner.py --wallet.name W --wallet.hotkey H serve --port 8089
  2) python bots/simple_signal_bot.py [--once] [--dry-run]

Env (optional):
  SN89_SERVE_URL   default http://127.0.0.1:8089/submit
  SN89_INTAKE_TOKEN  if you set one on serve
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

# SN89 board crypto pairs → Binance spot symbols (public, no API key).
PAIRS = {
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT",
    "XRPUSD": "XRPUSDT",
    "TAOUSD": "TAOUSDT",
}

SERVE_URL = os.getenv("SN89_SERVE_URL", "http://127.0.0.1:8089/submit")
INTAKE_TOKEN = os.getenv("SN89_INTAKE_TOKEN", "")
STATE_PATH = Path.home() / ".sn89" / "simple_bot_state.json"

# Mirror LF rules so we don't burn voided commits.
MAX_PER_UTC_DAY = 3
MIN_GAP_S = 3600

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "sn89-simple-bot/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def fetch_closes(symbol: str, interval: str = "15m", limit: int = 60) -> list[float]:
    url = f"{BINANCE_KLINES}?symbol={symbol}&interval={interval}&limit={limit}"
    rows = _get_json(url)
    return [float(r[4]) for r in rows]  # close


def sma(xs: list[float], n: int) -> float:
    return sum(xs[-n:]) / n


def decide(pair: str, closes: list[float]) -> str | None:
    """LONG if fast SMA > slow SMA, SHORT if below, else no trade.

    Toy rule only — replace with your real edge.
    """
    if len(closes) < 30:
        return None
    fast, slow = sma(closes, 8), sma(closes, 21)
    # Require a small separation so we don't flip on noise.
    edge = abs(fast - slow) / slow
    if edge < 0.0015:  # < 0.15%
        return None
    return "LONG" if fast > slow else "SHORT"


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"submits": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def can_submit(state: dict, now: float | None = None) -> tuple[bool, str]:
    now = now if now is not None else time.time()
    ts = [float(t) for t in state.get("submits", [])]
    ts = [t for t in ts if now - t < 2 * 86_400]
    if ts and now - max(ts) < MIN_GAP_S:
        wait = int(MIN_GAP_S - (now - max(ts)))
        return False, f"min gap: wait ~{wait}s"
    day = int(now // 86_400)
    today = sum(1 for t in ts if int(t // 86_400) == day)
    if today >= MAX_PER_UTC_DAY:
        return False, f"daily cap {MAX_PER_UTC_DAY}/UTC day reached"
    return True, "ok"


def post_signal(pair: str, direction: str, dry_run: bool) -> dict:
    body = {"trade_pair": pair, "direction": direction,
            "comment": "simple_signal_bot sma8/21"}
    if dry_run:
        return {"dry_run": True, **body}
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if INTAKE_TOKEN:
        headers["Authorization"] = f"Bearer {INTAKE_TOKEN}"
    req = urllib.request.Request(SERVE_URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return {"error": json.loads(raw), "http": e.code}
        except json.JSONDecodeError:
            return {"error": raw, "http": e.code}


def scan_once(dry_run: bool) -> int:
    ok, reason = can_submit(load_state())
    if not ok:
        print(f"skip submit: {reason}")
        return 0

    best: tuple[str, str, float] | None = None  # pair, dir, |fast-slow|/slow
    for pair, symbol in PAIRS.items():
        try:
            closes = fetch_closes(symbol)
        except Exception as e:  # noqa: BLE001
            print(f"{pair}: fetch failed: {e}")
            continue
        direction = decide(pair, closes)
        if not direction:
            print(f"{pair}: no signal")
            continue
        fast, slow = sma(closes, 8), sma(closes, 21)
        strength = abs(fast - slow) / slow
        print(f"{pair}: {direction} (strength={strength:.4%})")
        if best is None or strength > best[2]:
            best = (pair, direction, strength)

    if best is None:
        print("no pair with a clear signal")
        return 0

    pair, direction, _ = best
    print(f"→ submitting {pair} {direction} to {SERVE_URL}")
    res = post_signal(pair, direction, dry_run=dry_run)
    print(json.dumps(res, indent=2))

    if dry_run:
        return 0
    if res.get("ok") or res.get("commitment"):
        state = load_state()
        state["submits"] = state.get("submits", []) + [time.time()]
        save_state(state)
        return 0
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description="SN89 simple signal bot (template)")
    p.add_argument("--once", action="store_true", help="one scan then exit")
    p.add_argument("--dry-run", action="store_true",
                   help="print decision, do not POST (default if serve down)")
    p.add_argument("--interval", type=int, default=300,
                   help="seconds between scans when looping (default 300)")
    p.add_argument("--live", action="store_true",
                   help="actually POST to serve (without this, dry-run)")
    args = p.parse_args()

    dry = args.dry_run or not args.live
    if dry:
        print("DRY-RUN mode (pass --live to POST to serve)")

    if args.once:
        return scan_once(dry_run=dry)

    while True:
        scan_once(dry_run=dry)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
