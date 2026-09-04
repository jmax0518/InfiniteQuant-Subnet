#!/usr/bin/env python3
"""SN89 LLM signal bot — price features + news → OpenAI → local serve.

Designed for LF rules: ≤3 submits / UTC day, ≥1h gap. Typical schedule: 6
inference cycles/day; submit only when the model is confident.

Setup:
  export OPENAI_API_KEY=sk-...          # do NOT paste the key into chat
  export OPENAI_MODEL=gpt-4o            # or gpt-5.6-terra / gpt-5.6-sol / gpt-4o-mini
  # terminal 1:
  python neurons/miner.py --wallet.name W --wallet.hotkey H serve --port 8089
  # terminal 2:
  python bots/llm_signal_bot.py --once --dry-run
  python bots/llm_signal_bot.py --once --live

Not financial advice. Edge is not guaranteed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANDS_PATH = ROOT / "data" / "signals-bands.json"

# Board crypto pairs available on Binance spot.
PAIRS = {
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT",
    "XRPUSD": "XRPUSDT",
    "TAOUSD": "TAOUSDT",
}

SERVE_URL = os.getenv("SN89_SERVE_URL", "http://127.0.0.1:8089/submit")
INTAKE_TOKEN = os.getenv("SN89_INTAKE_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
MIN_CONFIDENCE = float(os.getenv("SN89_LLM_MIN_CONFIDENCE", "0.65"))

STATE_PATH = Path.home() / ".sn89" / "llm_bot_state.json"
MAX_PER_UTC_DAY = 3
MIN_GAP_S = 3600
HORIZON_H = 8

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
NEWS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]


def _http_get(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sn89-llm-bot/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_json(url: str, timeout: int = 25) -> object:
    return json.loads(_http_get(url, timeout=timeout).decode())


def load_bands() -> dict:
    raw = json.loads(BANDS_PATH.read_text(encoding="utf-8"))
    return raw.get("bands", raw)


def fetch_klines(symbol: str, interval: str = "15m", limit: int = 64) -> list[dict]:
    url = f"{BINANCE_KLINES}?symbol={symbol}&interval={interval}&limit={limit}"
    rows = _http_json(url)
    out = []
    for r in rows:
        out.append({
            "t": int(r[0]) // 1000,
            "o": float(r[1]),
            "h": float(r[2]),
            "l": float(r[3]),
            "c": float(r[4]),
            "v": float(r[5]),
        })
    return out


def atr_bps(bars: list[dict], n: int = 14) -> float:
    if len(bars) < n + 1:
        return 0.0
    trs = []
    for i in range(-n, 0):
        h, l, prev_c = bars[i]["h"], bars[i]["l"], bars[i - 1]["c"]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    mid = bars[-1]["c"]
    return (sum(trs) / len(trs)) / mid * 10_000 if mid else 0.0


def pct_change(bars: list[dict], lookback: int) -> float:
    if len(bars) <= lookback:
        return 0.0
    a, b = bars[-1 - lookback]["c"], bars[-1]["c"]
    return (b / a - 1.0) * 100.0 if a else 0.0


def compact_bars(bars: list[dict], last: int = 48) -> list[dict]:
    """Keep token cost down: round prices, drop volume noise digits."""
    slim = []
    for b in bars[-last:]:
        slim.append({
            "t": b["t"],
            "o": round(b["o"], 6),
            "h": round(b["h"], 6),
            "l": round(b["l"], 6),
            "c": round(b["c"], 6),
            "v": round(b["v"], 2),
        })
    return slim


def build_pair_features(pair: str, symbol: str, bands: dict) -> dict | None:
    try:
        bars_15 = fetch_klines(symbol, "15m", 64)
        bars_1h = fetch_klines(symbol, "1h", 48)
    except Exception as e:  # noqa: BLE001
        print(f"{pair}: kline fetch failed: {e}")
        return None
    band = bands.get(pair, {})
    tp = float(band.get("tp_bps", 0))
    atr15 = atr_bps(bars_15, 14)
    # How many ATRs to reach TP — high ratio = band may be hard (more washes).
    atr_to_tp = (tp / atr15) if atr15 > 0 else 999.0
    return {
        "trade_pair": pair,
        "tp_sl_bps": tp,
        "horizon_h": HORIZON_H,
        "last": bars_15[-1]["c"],
        "chg_1h_pct": round(pct_change(bars_15, 4), 3),    # 4×15m
        "chg_4h_pct": round(pct_change(bars_15, 16), 3),
        "chg_24h_pct": round(pct_change(bars_1h, 24), 3),
        "atr14_15m_bps": round(atr15, 1),
        "atr_to_tp_ratio": round(atr_to_tp, 2),
        "candles_15m": compact_bars(bars_15, 48),
        "candles_1h_tail": compact_bars(bars_1h, 24),
    }


def fetch_news(limit: int = 12) -> list[str]:
    headlines: list[str] = []
    for feed in NEWS_FEEDS:
        try:
            raw = _http_get(feed, timeout=15)
            root = ET.fromstring(raw)
            for item in root.iter():
                tag = item.tag.split("}")[-1].lower()
                if tag != "item":
                    continue
                title = None
                for child in item:
                    if child.tag.split("}")[-1].lower() == "title" and child.text:
                        title = child.text.strip()
                        break
                if title and title not in headlines:
                    headlines.append(title)
                if len(headlines) >= limit:
                    return headlines
        except Exception as e:  # noqa: BLE001
            print(f"news feed skip ({feed}): {e}")
    return headlines[:limit]


def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"submits": [], "inference_log": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # keep log bounded
    state["inference_log"] = state.get("inference_log", [])[-100:]
    state["submits"] = [t for t in state.get("submits", []) if time.time() - float(t) < 2 * 86_400]
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def can_submit(state: dict, now: float | None = None) -> tuple[bool, str]:
    now = now if now is not None else time.time()
    ts = [float(t) for t in state.get("submits", [])]
    if ts and now - max(ts) < MIN_GAP_S:
        return False, f"min gap: wait ~{int(MIN_GAP_S - (now - max(ts)))}s"
    day = int(now // 86_400)
    today = sum(1 for t in ts if int(t // 86_400) == day)
    if today >= MAX_PER_UTC_DAY:
        return False, f"daily cap {MAX_PER_UTC_DAY}/UTC day reached"
    return True, "ok"


SYSTEM_PROMPT = """You are a short-horizon crypto signal engine for Bittensor SN89.

Grading rules (critical):
- You only choose trade_pair + LONG or SHORT (or NONE).
- Each pair has a FIXED symmetric TP/SL in bps. First touch of TP = WON, SL = LOST.
- Crypto horizon is about 8 hours. Washes (neither side hit) do not help emissions.
- Hit-rate matters more than volume. Prefer NONE unless you expect TP before SL.
- Max 3 submits/UTC day for the miner; be selective.

Output STRICT JSON only:
{
  "action": "SUBMIT" | "NONE",
  "trade_pair": "BTCUSD" | ... | null,
  "direction": "LONG" | "SHORT" | null,
  "confidence": 0.0-1.0,
  "thesis": "one short sentence",
  "risks": "one short sentence"
}
"""


def build_user_payload(features: list[dict], news: list[str], submit_ok: bool, reason: str) -> str:
    # Drop bulky candle arrays from the "summary" view for ranking hint; full
    # candles stay in features for the model but we already capped length.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    slim = []
    for f in features:
        slim.append({k: v for k, v in f.items()})
    return json.dumps({
        "utc_now": now,
        "miner_can_submit_now": submit_ok,
        "miner_submit_block_reason": None if submit_ok else reason,
        "allowed_pairs": sorted(PAIRS.keys()),
        "news_headlines_24hish": news,
        "pairs": slim,
        "instruction": (
            "Pick at most ONE best setup with confidence>="
            f"{MIN_CONFIDENCE}, else action=NONE. "
            "Favor pairs where atr_to_tp_ratio is not extremely high "
            "(band reachable in ~8h). Align with news only if price agrees."
        ),
    }, separators=(",", ":"))


def call_openai(system: str, user: str) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Run:  export OPENAI_API_KEY=sk-...  "
            "(keep the key in your shell / .env — do not paste it into chat)"
        )
    body = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        OPENAI_URL,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")
        raise RuntimeError(f"OpenAI HTTP {e.code}: {err}") from e

    content = raw["choices"][0]["message"]["content"]
    usage = raw.get("usage", {})
    parsed = json.loads(content)
    parsed["_usage"] = usage
    parsed["_model"] = OPENAI_MODEL
    return parsed


def validate_decision(dec: dict, bands: dict) -> tuple[str, str, float] | None:
    """Return (pair, direction, confidence) or None."""
    action = str(dec.get("action", "NONE")).upper()
    if action != "SUBMIT":
        return None
    pair = str(dec.get("trade_pair") or "").upper()
    direction = str(dec.get("direction") or "").upper()
    try:
        conf = float(dec.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    if pair not in PAIRS or pair not in bands:
        print(f"reject: pair {pair!r} not on crypto board map")
        return None
    if direction not in ("LONG", "SHORT"):
        print(f"reject: bad direction {direction!r}")
        return None
    if conf < MIN_CONFIDENCE:
        print(f"reject: confidence {conf:.2f} < {MIN_CONFIDENCE}")
        return None
    return pair, direction, conf


def post_signal(pair: str, direction: str, thesis: str, dry_run: bool) -> dict:
    comment = re.sub(r"\s+", " ", (thesis or "")[:240])
    body = {"trade_pair": pair, "direction": direction, "comment": comment}
    if dry_run:
        return {"dry_run": True, **body}
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if INTAKE_TOKEN:
        headers["Authorization"] = f"Bearer {INTAKE_TOKEN}"
    req = urllib.request.Request(SERVE_URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return {"error": json.loads(raw), "http": e.code}
        except json.JSONDecodeError:
            return {"error": raw, "http": e.code}


def estimate_cost_usd(usage: dict, model: str) -> float | None:
    """Rough USD estimate from public list prices (short-context)."""
    # input, output per 1M tokens
    rates = {
        "gpt-5.6-sol": (5.0, 30.0),
        "gpt-5.6-terra": (2.0, 12.0),
        "gpt-5.6-luna": (0.20, 1.20),
        "gpt-4o": (2.50, 10.0),
        "gpt-4o-mini": (0.15, 0.60),
    }
    key = model.lower()
    if key not in rates:
        return None
    inn, out = rates[key]
    pt = float(usage.get("prompt_tokens") or 0)
    ct = float(usage.get("completion_tokens") or 0)
    return (pt * inn + ct * out) / 1_000_000


def run_cycle(dry_run: bool) -> int:
    bands = load_bands()
    state = load_state()
    submit_ok, reason = can_submit(state)

    print(f"model={OPENAI_MODEL}  serve={SERVE_URL}  min_conf={MIN_CONFIDENCE}")
    print(f"submit_allowed={submit_ok}" + (f" ({reason})" if not submit_ok else ""))

    features: list[dict] = []
    for pair, symbol in PAIRS.items():
        feat = build_pair_features(pair, symbol, bands)
        if feat:
            features.append(feat)
            print(
                f"  {pair}: last={feat['last']}  "
                f"1h={feat['chg_1h_pct']}% 4h={feat['chg_4h_pct']}%  "
                f"atr={feat['atr14_15m_bps']}bps  atr/tp={feat['atr_to_tp_ratio']}"
            )
    if not features:
        print("no market features — abort")
        return 1

    news = fetch_news(12)
    print(f"news headlines: {len(news)}")
    for h in news[:5]:
        print(f"  - {h}")

    user = build_user_payload(features, news, submit_ok, reason)
    # rough token hint
    print(f"prompt chars≈{len(SYSTEM_PROMPT) + len(user)}")

    try:
        dec = call_openai(SYSTEM_PROMPT, user)
    except Exception as e:  # noqa: BLE001
        print(f"LLM error: {e}")
        return 1

    usage = dec.pop("_usage", {})
    model = dec.pop("_model", OPENAI_MODEL)
    cost = estimate_cost_usd(usage, model)
    print("LLM decision:")
    print(json.dumps(dec, indent=2))
    if usage:
        msg = f"tokens: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}"
        if cost is not None:
            msg += f"  est_cost≈${cost:.4f}"
        print(msg)

    state.setdefault("inference_log", []).append({
        "ts": time.time(),
        "decision": dec,
        "usage": usage,
        "model": model,
    })

    picked = validate_decision(dec, bands)
    if not picked:
        print("→ no submit (NONE / low confidence / invalid)")
        save_state(state)
        return 0

    if not submit_ok:
        print(f"→ model wanted SUBMIT but miner blocked: {reason}")
        save_state(state)
        return 0

    pair, direction, conf = picked
    thesis = str(dec.get("thesis") or "")
    print(f"→ {'DRY-RUN ' if dry_run else ''}submit {pair} {direction} conf={conf:.2f}")
    res = post_signal(pair, direction, thesis, dry_run=dry_run)
    print(json.dumps(res, indent=2))

    if not dry_run and (res.get("ok") or res.get("commitment")):
        state.setdefault("submits", []).append(time.time())
    save_state(state)
    return 0 if dry_run or res.get("ok") or res.get("commitment") or res.get("dry_run") else 1


def main() -> int:
    p = argparse.ArgumentParser(description="SN89 LLM signal bot")
    p.add_argument("--once", action="store_true", help="one inference cycle then exit")
    p.add_argument("--live", action="store_true",
                   help="POST to serve when model submits (default is dry-run)")
    p.add_argument("--dry-run", action="store_true", help="force dry-run")
    p.add_argument("--interval", type=int, default=14400,
                   help="seconds between cycles when looping (default 14400 = 4h ≈ 6/day)")
    p.add_argument("--min-confidence", type=float, default=None,
                   help="override SN89_LLM_MIN_CONFIDENCE")
    args = p.parse_args()

    global MIN_CONFIDENCE
    if args.min_confidence is not None:
        MIN_CONFIDENCE = args.min_confidence

    dry = True if args.dry_run or not args.live else False
    if dry:
        print("DRY-RUN (pass --live to POST to miner serve)")

    if not OPENAI_API_KEY:
        print(
            "Missing OPENAI_API_KEY.\n"
            "  export OPENAI_API_KEY='sk-...'\n"
            "Do not paste the key into this chat — keep it in your environment only."
        )
        return 2

    if args.once:
        return run_cycle(dry_run=dry)

    while True:
        run_cycle(dry_run=dry)
        print(f"sleep {args.interval}s …")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
