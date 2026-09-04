#!/usr/bin/env python3
"""SN89 AI finance bot — ranked market+news pipeline → OpenAI → local serve.

Architecture (selective, cost-aware, SN89-native):
  1) Pull Binance candles for board crypto pairs
  2) Engineer features (trend, RSI, ATR vs TP band, swings)
  3) Pull headlines; score relevance / urgency per coin
  4) Pre-rank pairs; send only top-K to the LLM
  5) LLM returns SUBMIT/NONE + confidence
  6) Hard gates (confidence, tech/news agreement, 3/day, 1h gap)
  7) POST to miner `serve` only when --live

This is a high-quality *decision pipeline*, not a guaranteed edge.
Not financial advice.

Setup:
  export OPENAI_API_KEY=sk-...     # never paste into chat
  export OPENAI_MODEL=gpt-4o       # or gpt-5.6-terra / gpt-5.6-sol / gpt-4o-mini

  # terminal 1
  python neurons/miner.py --wallet.name W --wallet.hotkey H serve --port 8089
  # terminal 2
  python bots/ai_finance_bot.py --once --dry-run
  python bots/ai_finance_bot.py --once --live
  python bots/ai_finance_bot.py --live --interval 14400   # ~6 cycles/day
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANDS_PATH = ROOT / "data" / "signals-bands.json"
BOT_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Does not override existing exports."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(BOT_DIR / ".env")
_load_dotenv(Path.home() / ".sn89" / "ai_finance_bot.env")

PAIRS = {
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT",
    "XRPUSD": "XRPUSDT",
    "TAOUSD": "TAOUSDT",
}

# Keyword bags for news→coin matching (lowercase substrings / word-ish tokens).
COIN_KEYWORDS = {
    "BTCUSD": [
        "bitcoin", "btc", "microstrategy", "strategy Inc", "spot bitcoin",
        "bitcoin etf", "btc etf", "bitcoin miner", "el salvador",
    ],
    "ETHUSD": [
        "ethereum", " ether", "eth ", "eth,", "eth-", "$eth", "vitalik",
        "ether etf", "eth etf", "ethereum etf", "staking to ethereum",
        "ethereum staking", "sec ethereum",
    ],
    "SOLUSD": [
        "solana", "sol ", " sol,", "$sol", "solana etf", "moneygram",
    ],
    "XRPUSD": [
        "xrp", "ripple", "sec ripple", "xrp etf", "garlinghouse",
    ],
    "TAOUSD": [
        "bittensor", " $tao", "tao ", "tao,", "opulent", "subnet",
    ],
}
MACRO_KEYWORDS = [
    "fed", "fomc", "cpi", "inflation", "rate cut", "rate hike", "rate pause",
    "sec ", " etf", "etf ", "hack", "exploit", "liquidation", "ban", "lawsuit",
    "war", "treasury", "recession", "jobs report", "powell", "stablecoin",
    "mica", "interest rate",
]
BULLISH_HINTS = [
    "approval", "approved", "etf inflow", "inflows", "all-time high", "ath",
    "surge", "rally", "rallies", "soar", "jumps", "jump ", "eyes $", "partnership",
    "upgrade", "launch", "record inflow", "staking", "relief", "growth",
    "expands", "integration", "adoption", "accumulate", "buys ", "bought",
]
BEARISH_HINTS = [
    "hack", "exploit", "sec charges", "lawsuit", "ban", "crash", "outflow",
    "outflows", "arrest", "fraud", "insolvent", "default", "sell-off", "selloff",
    "dump", "plunge", "slides", "slide ", "lay off", "layoff", "cuts of staff",
    "staff cut", "fake crypto", "fooled", "clash over",
]

SERVE_URL = os.getenv("SN89_SERVE_URL", "http://127.0.0.1:8089/submit")
INTAKE_TOKEN = os.getenv("SN89_INTAKE_TOKEN", "")
# auto = HTTP serve first, then CLI direct; http | direct force one path
SUBMIT_MODE = os.getenv("SN89_SUBMIT_MODE", "auto").strip().lower()
MINER_ROOT = Path(os.getenv("SN89_MINER_ROOT", "/root/MVTRX_08_05/InfiniteQuant-Subnet"))
MINER_PYTHON = os.getenv(
    "SN89_MINER_PYTHON",
    str(MINER_ROOT / ".venv" / "bin" / "python"),
)
WALLET_NAME = os.getenv("WALLET_NAME", "GOLD")
# Own miner hotkey (registered SN89). Never default to friend hotkey "sn89".
WALLET_HOTKEY = os.getenv("WALLET_HOTKEY", "iq89")
# Prefer explicit ss58; else resolve from wallet file if present
HOTKEY_SS58 = os.getenv(
    "SN89_HOTKEY_SS58",
    "5Co94YcY8EDTAHcgFs5sB4dW4R999urfMDExXhAJxysk2gU8",
).strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")


def _default_api_url(key: str) -> str:
    """OpenRouter keys are sk-or-v1-…; OpenAI keys are sk-…"""
    if os.getenv("OPENAI_API_URL"):
        return os.environ["OPENAI_API_URL"]
    if key.startswith("sk-or-"):
        return "https://openrouter.ai/api/v1/chat/completions"
    return "https://api.openai.com/v1/chat/completions"


def _default_model(key: str) -> str:
    if os.getenv("OPENAI_MODEL"):
        return os.environ["OPENAI_MODEL"]
    # OpenRouter wants provider/model ids
    if key.startswith("sk-or-"):
        return "openai/gpt-4o"
    return "gpt-4o"


OPENAI_URL = _default_api_url(OPENAI_API_KEY)
OPENAI_MODEL = _default_model(OPENAI_API_KEY)

MIN_CONFIDENCE = float(os.getenv("SN89_LLM_MIN_CONFIDENCE", "0.68"))
TOP_K = int(os.getenv("SN89_BOT_TOP_K", "2"))
MIN_PRE_SCORE = float(os.getenv("SN89_BOT_MIN_PRE_SCORE", "0.35"))
REQUIRE_TECH_ALIGN = os.getenv("SN89_BOT_REQUIRE_TECH_ALIGN", "1") == "1"
DAILY_BUDGET_USD = float(os.getenv("SN89_LLM_DAILY_BUDGET_USD", "1.0"))
MAX_CYCLES_PER_UTC_DAY = int(os.getenv("SN89_BOT_MAX_CYCLES_PER_DAY", "6"))

STATE_PATH = Path.home() / ".sn89" / "ai_finance_bot_state.json"
MAX_PER_UTC_DAY = 3
MIN_GAP_S = 3600
HORIZON_H = 8

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
NEWS_FEEDS = [
    # CoinDesk often 308-redirects; _http_get follows 301/302/307/308.
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://www.theblock.co/rss.xml",
]


# ── HTTP ─────────────────────────────────────────────────────────────────────
class _RedirectAll(urllib.request.HTTPRedirectHandler):
    """Follow 308/307 as well (stdlib historically skipped some)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if code in (301, 302, 303, 307, 308):
            return urllib.request.Request(
                newurl,
                headers={k: v for k, v in req.headers.items()},
                method="GET" if code in (301, 302, 303, 308) else req.get_method(),
            )
        return None


_URL_OPENER = urllib.request.build_opener(_RedirectAll)


def _http_get(url: str, timeout: int = 25, max_hops: int = 5) -> bytes:
    """GET with User-Agent + manual 308-safe redirects."""
    seen: set[str] = set()
    cur = url
    for _ in range(max_hops):
        if cur in seen:
            break
        seen.add(cur)
        req = urllib.request.Request(
            cur,
            headers={
                "User-Agent": "sn89-ai-finance-bot/2.1",
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
            },
        )
        try:
            with _URL_OPENER.open(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            # Some stacks surface 308 as HTTPError instead of redirecting.
            if e.code in (301, 302, 303, 307, 308):
                loc = e.headers.get("Location")
                if not loc:
                    raise
                cur = urllib.parse.urljoin(cur, loc)
                continue
            raise
    raise RuntimeError(f"redirect loop or failed GET: {url}")


def _http_json(url: str, timeout: int = 25):
    return json.loads(_http_get(url, timeout=timeout).decode())


# ── Market features ──────────────────────────────────────────────────────────
def load_bands() -> dict:
    raw = json.loads(BANDS_PATH.read_text(encoding="utf-8"))
    return raw.get("bands", raw)


def fetch_klines(symbol: str, interval: str, limit: int) -> list[dict]:
    url = f"{BINANCE_KLINES}?symbol={symbol}&interval={interval}&limit={limit}"
    rows = _http_json(url)
    return [{
        "t": int(r[0]) // 1000,
        "o": float(r[1]), "h": float(r[2]), "l": float(r[3]),
        "c": float(r[4]), "v": float(r[5]),
    } for r in rows]


def ema(xs: list[float], n: int) -> float:
    if not xs:
        return 0.0
    k = 2 / (n + 1)
    e = xs[0]
    for x in xs[1:]:
        e = x * k + e * (1 - k)
    return e


def rsi(closes: list[float], n: int = 14) -> float:
    if len(closes) <= n:
        return 50.0
    gains, losses = [], []
    for i in range(-n, 0):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag, al = sum(gains) / n, sum(losses) / n
    if al == 0:
        return 100.0
    rs = ag / al
    return 100 - (100 / (1 + rs))


def atr_bps(bars: list[dict], n: int = 14) -> float:
    if len(bars) < n + 1:
        return 0.0
    trs = []
    for i in range(-n, 0):
        h, l, pc = bars[i]["h"], bars[i]["l"], bars[i - 1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    mid = bars[-1]["c"]
    return (sum(trs) / len(trs)) / mid * 10_000 if mid else 0.0


def pct(bars: list[dict], lookback: int) -> float:
    if len(bars) <= lookback:
        return 0.0
    a, b = bars[-1 - lookback]["c"], bars[-1]["c"]
    return (b / a - 1.0) * 100.0 if a else 0.0


def swing_pos(bars: list[dict], n: int = 32) -> float:
    """0 = at local low, 1 = at local high over last n bars."""
    window = bars[-n:]
    lo = min(b["l"] for b in window)
    hi = max(b["h"] for b in window)
    if hi <= lo:
        return 0.5
    return (bars[-1]["c"] - lo) / (hi - lo)


@dataclass
class PairSnapshot:
    trade_pair: str
    tp_sl_bps: float
    last: float
    chg_1h_pct: float
    chg_4h_pct: float
    chg_24h_pct: float
    atr14_15m_bps: float
    atr_to_tp_ratio: float
    rsi14: float
    ema_fast: float
    ema_slow: float
    trend: str                 # UP / DOWN / FLAT
    swing_pos: float
    tech_bias: str             # LONG / SHORT / NONE
    tech_score: float          # 0..1 strength of technical setup
    reach_score: float         # 0..1 how reachable TP looks in ~8h
    news_score: float = 0.0
    news_bias: str = "NONE"
    matched_headlines: list[str] = field(default_factory=list)
    pre_score: float = 0.0
    candles_15m: list[dict] = field(default_factory=list)
    tech_components: dict = field(default_factory=dict)

    def llm_view(self) -> dict:
        """Compact view for the model (token-efficient).

        Heuristic tech_* fields are hints only — the LLM should compute its own
        tech_bias / tech_score from candles + the FULL news list in the payload.
        """
        return {
            "trade_pair": self.trade_pair,
            "tp_sl_bps": self.tp_sl_bps,
            "horizon_h": HORIZON_H,
            "last": self.last,
            "chg_1h_pct": self.chg_1h_pct,
            "chg_4h_pct": self.chg_4h_pct,
            "chg_24h_pct": self.chg_24h_pct,
            "atr14_15m_bps": self.atr14_15m_bps,
            "atr_to_tp_ratio": self.atr_to_tp_ratio,
            "rsi14": round(self.rsi14, 1),
            "trend": self.trend,
            "swing_pos": round(self.swing_pos, 3),
            # Local heuristic (optional prior — do NOT copy blindly)
            "heuristic_tech_bias": self.tech_bias,
            "heuristic_tech_score": round(self.tech_score, 3),
            "heuristic_tech_components": self.tech_components,
            "reach_score": round(self.reach_score, 3),
            "heuristic_news_bias": self.news_bias,
            "heuristic_news_score": round(self.news_score, 3),
            "matched_headlines": self.matched_headlines[:5],
            "pre_score": round(self.pre_score, 3),
            "candles_15m_tail": self.candles_15m[-32:],
        }


def headlines_for_llm(headlines: list["Headline"], limit: int = 40) -> list[dict]:
    """Full news feed for the model (title + tone + urgency)."""
    return [
        {"title": h.title, "tone": h.tone, "urgency": round(float(h.urgency), 2)}
        for h in headlines[:limit]
    ]


def _trend_from_closes(closes: list[float], fast_n: int = 8, slow_n: int = 21
                       ) -> tuple[str, float]:
    """Return (UP|DOWN|FLAT, gap_frac)."""
    if len(closes) < slow_n + 2:
        return "FLAT", 0.0
    fast, slow = ema(closes, fast_n), ema(closes, slow_n)
    gap = (fast - slow) / slow if slow else 0.0
    if gap > 0.0010:
        return "UP", gap
    if gap < -0.0010:
        return "DOWN", gap
    return "FLAT", gap


def _efficiency_ratio(closes: list[float], n: int = 16) -> float:
    """Kaufman ER: 1 = clean trend, 0 = pure chop — critical for fixed-band SN89."""
    if len(closes) <= n:
        return 0.0
    window = closes[-(n + 1):]
    net = abs(window[-1] - window[0])
    path = sum(abs(window[i] - window[i - 1]) for i in range(1, len(window)))
    return (net / path) if path > 0 else 0.0


def _directional_persistence(closes: list[float], n: int = 12) -> float:
    """Share of bars closing up minus share closing down, in [-1, 1]."""
    if len(closes) <= n:
        return 0.0
    ups = downs = 0
    for i in range(-n, 0):
        if closes[i] > closes[i - 1]:
            ups += 1
        elif closes[i] < closes[i - 1]:
            downs += 1
    tot = ups + downs
    return ((ups - downs) / tot) if tot else 0.0


def _volume_confirm(bars: list[dict], direction: str, n: int = 16) -> float:
    """0..1: is recent volume elevated on bars in the signal direction?"""
    if len(bars) < n + 2 or direction not in ("LONG", "SHORT"):
        return 0.5
    window = bars[-n:]
    vols = [b["v"] for b in window]
    base = sorted(vols)[len(vols) // 2] or 1.0
    directed = []
    for b in window:
        up = b["c"] >= b["o"]
        if direction == "LONG" and up:
            directed.append(b["v"])
        if direction == "SHORT" and not up:
            directed.append(b["v"])
    if not directed:
        return 0.25
    ratio = (sum(directed) / len(directed)) / base
    # map ~0.7..1.8 → 0..1
    return max(0.0, min(1.0, (ratio - 0.7) / 1.1))


def _structure_bias(bars: list[dict], lookback: int = 24) -> tuple[str, float]:
    """Higher-highs/higher-lows vs lower-highs/lower-lows on two halves."""
    if len(bars) < lookback:
        return "FLAT", 0.0
    w = bars[-lookback:]
    mid = len(w) // 2
    a, b = w[:mid], w[mid:]
    ah, al = max(x["h"] for x in a), min(x["l"] for x in a)
    bh, bl = max(x["h"] for x in b), min(x["l"] for x in b)
    if bh > ah and bl > al:
        return "UP", 1.0
    if bh < ah and bl < al:
        return "DOWN", 1.0
    if bh > ah or bl > al:
        return "UP", 0.45
    if bh < ah or bl < al:
        return "DOWN", 0.45
    return "FLAT", 0.0


def compute_tech_edge(bars_15: list[dict], bars_1h: list[dict], tp_bps: float
                      ) -> tuple[str, float, str, float, float, dict]:
    """SN89-oriented technical edge.

    Design goals for fixed 1R / ~8h grading:
      1) Multi-timeframe agreement (15m + 1h) — strongest filter
      2) Trend + pullback (not chase extremes) — better TP-before-SL odds
      3) Efficiency ratio — kill chop (washes)
      4) Persistence + volume — confirm real pressure
      5) Extension penalty — already traveled toward TP → lower edge

    Returns: tech_bias, tech_score, trend_15m, rsi14, swing_pos, components
    """
    closes = [b["c"] for b in bars_15]
    closes_1h = [b["c"] for b in bars_1h]
    r = rsi(closes, 14)
    sp = swing_pos(bars_15, 32)
    chg1 = pct(bars_15, 4)     # ~1h
    chg4 = pct(bars_15, 16)    # ~4h
    er = _efficiency_ratio(closes, 16)
    persist = _directional_persistence(closes, 12)
    t15, gap15 = _trend_from_closes(closes[-48:])
    t1h, gap1h = _trend_from_closes(closes_1h[-36:])
    struct, struct_s = _structure_bias(bars_15, 24)

    # ── side scores in [-1, +1] space then map to bias/score ─────────────
    long_s = short_s = 0.0
    comps: dict = {
        "er": round(er, 3),
        "persist": round(persist, 3),
        "trend_15m": t15,
        "trend_1h": t1h,
        "structure": struct,
        "regime": "chop",
    }

    # (1) MTF trend — core
    if t15 == "UP":
        long_s += 0.22
    elif t15 == "DOWN":
        short_s += 0.22
    if t1h == "UP":
        long_s += 0.28
    elif t1h == "DOWN":
        short_s += 0.28
    if t15 == t1h == "UP":
        long_s += 0.18
        comps["mtf"] = "aligned_up"
    elif t15 == t1h == "DOWN":
        short_s += 0.18
        comps["mtf"] = "aligned_down"
    elif t15 != "FLAT" and t1h != "FLAT" and t15 != t1h:
        # conflict — damp both later
        comps["mtf"] = "conflict"
        long_s *= 0.55
        short_s *= 0.55
    else:
        comps["mtf"] = "mixed"

    # (2) Structure
    if struct == "UP":
        long_s += 0.12 * struct_s
    elif struct == "DOWN":
        short_s += 0.12 * struct_s

    # (3) Pullback-in-trend (preferred SN89 entry) vs chase / mean-revert
    if t15 == "UP" or t1h == "UP":
        if 38 <= r <= 55 and sp < 0.55:
            long_s += 0.20          # dip buy in uptrend
            comps["entry_style"] = "pullback_long"
        elif 55 < r <= 68 and persist > 0.15:
            long_s += 0.10          # continuation
            comps["entry_style"] = "cont_long"
        elif r > 72:
            long_s -= 0.18          # extended — bad for fixed TP chase
            short_s += 0.08          # mild mean-revert pressure
            comps["entry_style"] = "extended_long"
    if t15 == "DOWN" or t1h == "DOWN":
        if 45 <= r <= 62 and sp > 0.45:
            short_s += 0.20         # bounce sell in downtrend
            comps["entry_style"] = "pullback_short"
        elif 32 <= r < 45 and persist < -0.15:
            short_s += 0.10
            comps["entry_style"] = "cont_short"
        elif r < 28:
            short_s -= 0.18
            long_s += 0.08
            comps["entry_style"] = "extended_short"

    # (4) Momentum agreement with side
    if chg1 > 0.12 and chg4 > 0:
        long_s += 0.10
    if chg1 < -0.12 and chg4 < 0:
        short_s += 0.10
    if persist > 0.25:
        long_s += 0.08
    if persist < -0.25:
        short_s += 0.08

    # (5) Efficiency — chop kills edge (washes on SN89)
    if er >= 0.35:
        comps["regime"] = "trend"
        scale = 0.85 + 0.35 * min(1.0, (er - 0.35) / 0.35)
    elif er >= 0.22:
        comps["regime"] = "mild_trend"
        scale = 0.70
    else:
        comps["regime"] = "chop"
        scale = 0.40
    long_s *= scale
    short_s *= scale

    # (6) Extension vs band: if 4h move already ≥ ~60% of TP in that direction,
    # remaining room to TP is small → cut score (asymmetric trap into SL).
    if tp_bps > 0:
        moved_bps = abs(chg4) * 100  # chg4 is percent
        ext = moved_bps / tp_bps
        comps["extension_vs_tp"] = round(ext, 2)
        if ext >= 0.60:
            if chg4 > 0:
                long_s *= 0.55
            elif chg4 < 0:
                short_s *= 0.55

    # Decide side
    edge = long_s - short_s
    if edge >= 0.18:
        bias = "LONG"
        raw = long_s
    elif edge <= -0.18:
        bias = "SHORT"
        raw = short_s
    else:
        bias = "NONE"
        raw = max(long_s, short_s) * 0.45

    # Volume confirm only strengthens an existing bias
    vol_c = _volume_confirm(bars_15, bias if bias != "NONE" else "LONG")
    if bias != "NONE":
        raw *= 0.75 + 0.35 * vol_c
    comps["volume_confirm"] = round(vol_c, 3)
    comps["long_s"] = round(long_s, 3)
    comps["short_s"] = round(short_s, 3)
    comps["edge"] = round(edge, 3)

    # Calibrate to 0..1 (empirical soft cap ~1.2 before clamp)
    score = max(0.0, min(1.0, raw / 1.05))
    if bias == "NONE":
        score = min(score, 0.45)

    return bias, score, t15, r, sp, comps


def build_snapshot(pair: str, symbol: str, bands: dict) -> PairSnapshot | None:
    try:
        bars = fetch_klines(symbol, "15m", 80)
        bars_1h = fetch_klines(symbol, "1h", 48)
    except Exception as e:  # noqa: BLE001
        print(f"{pair}: market fetch failed: {e}")
        return None

    closes = [b["c"] for b in bars]
    tp = float(bands.get(pair, {}).get("tp_bps", 0))
    atr = atr_bps(bars, 14)
    atr_to_tp = (tp / atr) if atr > 0 else 99.0
    # Reachability: best when ~0.6–1.8 ATR-to-TP (band hittable, not noise).
    if atr_to_tp <= 0:
        reach = 0.0
    elif atr_to_tp < 0.5:
        reach = 0.45  # very noisy vs band
    elif atr_to_tp <= 1.8:
        reach = 1.0 - abs(atr_to_tp - 1.0) * 0.25
    elif atr_to_tp <= 3.0:
        reach = 0.55
    else:
        reach = max(0.15, 1.2 / atr_to_tp)

    fast, slow = ema(closes[-40:], 8), ema(closes[-40:], 21)
    chg1, chg4 = pct(bars, 4), pct(bars, 16)
    chg24 = pct(bars_1h, 24)

    tech_bias, tech_score, trend, r, sp, comps = compute_tech_edge(
        bars, bars_1h, tp
    )

    slim = [{
        "t": b["t"],
        "o": round(b["o"], 6), "h": round(b["h"], 6),
        "l": round(b["l"], 6), "c": round(b["c"], 6),
        "v": round(b["v"], 2),
    } for b in bars[-48:]]

    return PairSnapshot(
        trade_pair=pair, tp_sl_bps=tp, last=bars[-1]["c"],
        chg_1h_pct=round(chg1, 3), chg_4h_pct=round(chg4, 3),
        chg_24h_pct=round(chg24, 3),
        atr14_15m_bps=round(atr, 1), atr_to_tp_ratio=round(atr_to_tp, 2),
        rsi14=r, ema_fast=fast, ema_slow=slow, trend=trend,
        swing_pos=sp, tech_bias=tech_bias, tech_score=tech_score,
        reach_score=round(min(1.0, max(0.0, reach)), 3),
        candles_15m=slim,
        tech_components=comps,
    )


# ── News ─────────────────────────────────────────────────────────────────────
@dataclass
class Headline:
    title: str
    urgency: float
    tone: str  # BULL / BEAR / NEUTRAL


def _tone(title: str) -> tuple[str, float]:
    t = title.lower()
    bull = sum(1 for k in BULLISH_HINTS if k in t)
    bear = sum(1 for k in BEARISH_HINTS if k in t)
    urg = 0.35
    if any(k in t for k in MACRO_KEYWORDS):
        urg += 0.35
    if bull or bear:
        urg += 0.2
    urg = min(1.0, urg)
    if bear > bull:
        return "BEAR", urg
    if bull > bear:
        return "BULL", urg
    return "NEUTRAL", urg


def fetch_news(limit: int = 20) -> list[Headline]:
    seen: set[str] = set()
    out: list[Headline] = []
    for feed in NEWS_FEEDS:
        try:
            root = ET.fromstring(_http_get(feed, timeout=15))
        except Exception as e:  # noqa: BLE001
            print(f"news skip {feed}: {e}")
            continue
        for item in root.iter():
            if item.tag.split("}")[-1].lower() != "item":
                continue
            title = None
            for child in item:
                if child.tag.split("}")[-1].lower() == "title" and child.text:
                    title = child.text.strip()
                    break
            if not title or title in seen:
                continue
            seen.add(title)
            tone, urg = _tone(title)
            out.append(Headline(title=title, urgency=urg, tone=tone))
            if len(out) >= limit:
                return out
    return out


def _coin_match(title: str, keys: list[str]) -> bool:
    tl = title.lower()
    for k in keys:
        kl = k.lower()
        if kl.startswith("$"):
            if kl in tl:
                return True
            continue
        # word-ish: avoid matching tiny tokens inside unrelated words
        if len(kl.strip()) <= 3:
            if re.search(rf"(?<![a-z0-9]){re.escape(kl.strip())}(?![a-z0-9])", tl):
                return True
        elif kl in tl:
            return True
    return False


def attach_news(snap: PairSnapshot, headlines: list[Headline]) -> None:
    keys = COIN_KEYWORDS.get(snap.trade_pair, [])
    matched: list[Headline] = [h for h in headlines if _coin_match(h.title, keys)]
    macro_only = False
    # If no coin-specific hit, allow high-urgency macro to lightly affect BTC/ETH.
    if not matched and snap.trade_pair in ("BTCUSD", "ETHUSD"):
        matched = [h for h in headlines if h.urgency >= 0.65][:3]
        macro_only = bool(matched)

    if not matched:
        snap.news_score = 0.0
        snap.news_bias = "NONE"
        snap.matched_headlines = []
        return

    matched.sort(key=lambda h: (h.urgency, 0 if h.tone == "NEUTRAL" else 1), reverse=True)
    bull = sum(h.urgency for h in matched if h.tone == "BULL")
    bear = sum(h.urgency for h in matched if h.tone == "BEAR")
    neu = sum(h.urgency * 0.45 for h in matched if h.tone == "NEUTRAL")
    # Relevance always counts — so "has news" is visible even when tone is mixed.
    relevance = min(1.0, 0.22 * len(matched) + 0.12 * max(h.urgency for h in matched))
    if macro_only:
        relevance *= 0.55
    directional = abs(bull - bear)
    strength = min(1.0, (bull + bear + neu) / max(1.5, len(matched)))
    snap.news_score = min(1.0, 0.55 * relevance + 0.45 * max(strength, directional * 0.5))

    if bull - bear >= 0.2:
        snap.news_bias = "LONG"
    elif bear - bull >= 0.2:
        snap.news_bias = "SHORT"
    else:
        snap.news_bias = "NONE"

    # Annotate tone in the printed/LLM headline list: [BULL] title
    snap.matched_headlines = [
        f"[{h.tone}/{h.urgency:.2f}] {h.title}" for h in matched[:5]
    ]


def pre_score(snap: PairSnapshot) -> float:
    """Deterministic priority before spending LLM tokens."""
    tech = snap.tech_score if snap.tech_bias != "NONE" else snap.tech_score * 0.35
    news = snap.news_score
    # Agreement bonus
    agree = 0.0
    if snap.tech_bias != "NONE" and snap.tech_bias == snap.news_bias:
        agree = 0.25
    elif snap.news_bias == "NONE":
        agree = 0.05
    elif snap.tech_bias != "NONE" and snap.news_bias != snap.tech_bias:
        agree = -0.2  # conflict → deprioritize
    score = 0.50 * tech + 0.25 * news + 0.25 * snap.reach_score + agree
    return max(0.0, min(1.0, score))


# ── State / submit gates ─────────────────────────────────────────────────────
def load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"submits": [], "cycles": []}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["cycles"] = state.get("cycles", [])[-80:]
    now = time.time()
    state["submits"] = [t for t in state.get("submits", []) if now - float(t) < 2 * 86_400]
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _ss58_from_text(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if raw.startswith("{"):
        try:
            return str(json.loads(raw).get("ss58Address") or "").strip()
        except (json.JSONDecodeError, TypeError):
            return ""
    # plain ss58 line
    return raw.split()[0].strip()


def _resolve_hotkey_ss58() -> str:
    if HOTKEY_SS58:
        return _ss58_from_text(HOTKEY_SS58)
    pub = Path.home() / ".bittensor" / "wallets" / WALLET_NAME / "hotkeys" / f"{WALLET_HOTKEY}pub.txt"
    if pub.exists():
        ss58 = _ss58_from_text(pub.read_text(encoding="utf-8"))
        if ss58:
            return ss58
    hk = Path.home() / ".bittensor" / "wallets" / WALLET_NAME / "hotkeys" / WALLET_HOTKEY
    if hk.exists():
        try:
            return str(json.loads(hk.read_text(encoding="utf-8")).get("ss58Address") or "")
        except (json.JSONDecodeError, OSError, TypeError):
            return ""
    return ""


def miner_submit_timestamps(hotkey: str | None = None) -> list[float]:
    """Read the same local submit log the miner serve/CLI uses (~/.sn89/submits_<hk>.json)."""
    hk = hotkey or _resolve_hotkey_ss58()
    if not hk:
        return []
    path = Path.home() / ".sn89" / f"submits_{hk}.json"
    try:
        return [float(t) for t in json.loads(path.read_text(encoding="utf-8"))]
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError, OSError):
        return []


def _dedupe_submit_ts(ts: list[float], window_s: float = 30.0) -> list[float]:
    """Collapse near-duplicate timestamps (bot state + miner log often differ by ms)."""
    out: list[float] = []
    for t in sorted(float(x) for x in ts):
        if not out or t - out[-1] >= window_s:
            out.append(t)
    return out


def can_submit(state: dict) -> tuple[bool, str]:
    now = time.time()
    # Merge bot state + miner on-disk log so dashboard respects real SN89 quota.
    # Dedupe: the same commit is often recorded twice with microsecond skew.
    ts = _dedupe_submit_ts(
        [float(t) for t in state.get("submits", [])] + miner_submit_timestamps()
    )
    if ts and now - max(ts) < MIN_GAP_S:
        return False, f"min gap: wait ~{int(MIN_GAP_S - (now - max(ts)))}s"
    day = int(now // 86_400)
    if sum(1 for t in ts if int(t // 86_400) == day) >= MAX_PER_UTC_DAY:
        return False, f"daily cap {MAX_PER_UTC_DAY}/UTC day (resets 00:00 UTC)"
    return True, "ok"


# ── LLM ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a selective short-horizon crypto signal engine for Bittensor SN89.

Protocol facts:
- Submit only trade_pair + LONG/SHORT. TP/SL are FIXED symmetric bps per pair.
- First touch TP = WON, SL = LOST within ~8h. Washes do not help.
- Miner may submit at most 3 times per UTC day; hit-rate >> volume.
- YOU compute tech_bias/tech_score from candles (heuristic_* fields are optional priors only).
- YOU must read the FULL news list (all_news) — not only matched_headlines.
- Prefer NONE unless you believe TP is more likely than SL.

LONG REASONING: each pair needs a `reasoning` field of 4–8 sentences (structure → tech →
news from all_news → reach/wash → final call). tech_reason and news_reason must be
2+ full sentences. best.why must be a multi-sentence memo.

Return STRICT JSON:
{
  "scores": [
    {
      "trade_pair": "BTCUSD",
      "tech_bias": "LONG"|"SHORT"|"NONE",
      "tech_score": 0.0,
      "tech_reason": "2+ sentences",
      "news_bias": "LONG"|"SHORT"|"NONE",
      "news_reason": "2+ sentences",
      "direction": "LONG"|"SHORT"|"NONE",
      "confidence": 0.0,
      "thesis": "2+ sentences",
      "reasoning": "4-8 sentence analysis"
    }
  ],
  "best": {
    "action": "SUBMIT" | "NONE",
    "trade_pair": null,
    "direction": null,
    "confidence": 0.0,
    "why": "3-6 sentences"
  },
  "risks": "2+ sentences",
  "why_not_others": "2+ sentences"
}
Include every candidate pair in scores.
"""


def call_openai(candidates: list[PairSnapshot], all_news: list,
                submit_ok: bool, block_reason: str) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY not set. export OPENAI_API_KEY=sk-... "
            "(do not paste the key into chat)"
        )
    news_payload = (
        headlines_for_llm(all_news)
        if all_news and hasattr(all_news[0], "title")
        else all_news
    )
    payload = {
        "utc_now": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "miner_can_submit_now": submit_ok,
        "miner_block_reason": None if submit_ok else block_reason,
        "min_confidence_required": MIN_CONFIDENCE,
        "all_news": news_payload,
        "candidates": [c.llm_view() for c in candidates],
        "instruction": (
            "Using candles, compute your own tech_score per pair. "
            "Using ALL news in all_news, assess news impact per pair. "
            "Then choose at most ONE trade. "
            f"Only best.action=SUBMIT if confidence>={MIN_CONFIDENCE}."
        ),
    }
    body = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "max_tokens": int(os.getenv("SN89_LLM_MAX_TOKENS", "6000")),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    # OpenRouter optional rankings headers (harmless elsewhere).
    if "openrouter.ai" in OPENAI_URL:
        headers["HTTP-Referer"] = "https://github.com/DeltaCompute24/InfiniteQuant-Subnet"
        headers["X-Title"] = "SN89 AI finance bot"

    req = urllib.request.Request(
        OPENAI_URL,
        data=json.dumps(body).encode(),
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenAI HTTP {e.code}: {e.read().decode(errors='replace')}") from e

    content = raw["choices"][0]["message"]["content"]
    dec = json.loads(content)
    dec["_usage"] = raw.get("usage", {})
    dec["_model"] = OPENAI_MODEL
    return dec


def estimate_cost_usd(usage: dict, model: str) -> float | None:
    rates = {
        "gpt-5.6-sol": (5.0, 30.0),
        "openai/gpt-5.6-sol": (5.0, 30.0),
        "gpt-5.6-terra": (2.0, 12.0),
        "openai/gpt-5.6-terra": (2.0, 12.0),
        "gpt-5.6-luna": (0.20, 1.20),
        "openai/gpt-5.6-luna": (0.20, 1.20),
        "gpt-4o": (2.50, 10.0),
        "openai/gpt-4o": (2.50, 10.0),
        "gpt-4o-mini": (0.15, 0.60),
        "openai/gpt-4o-mini": (0.15, 0.60),
        "anthropic/claude-sonnet-4": (3.0, 15.0),
        "anthropic/claude-3.5-sonnet": (3.0, 15.0),
    }
    key = model.lower()
    if key not in rates:
        # fallback: treat unknown OpenRouter chat models like gpt-4o mid-tier
        if "/" in key:
            inn, out = 2.50, 10.0
        else:
            return None
    else:
        inn, out = rates[key]
    return (
        float(usage.get("prompt_tokens") or 0) * inn
        + float(usage.get("completion_tokens") or 0) * out
    ) / 1_000_000


def _utc_day(ts: float | None = None) -> int:
    return int((ts if ts is not None else time.time()) // 86_400)


def spent_today_usd(state: dict) -> float:
    day = _utc_day()
    total = 0.0
    for c in state.get("cycles", []):
        if _utc_day(float(c.get("ts", 0))) != day:
            continue
        total += float(c.get("est_cost_usd") or 0.0)
    return total


def cycles_today(state: dict) -> int:
    day = _utc_day()
    return sum(1 for c in state.get("cycles", []) if _utc_day(float(c.get("ts", 0))) == day)


# ── Decision gates ───────────────────────────────────────────────────────────
def gate_decision(dec: dict, candidates: list[PairSnapshot]) -> tuple[str, str, float] | None:
    if str(dec.get("action", "NONE")).upper() != "SUBMIT":
        return None
    pair = str(dec.get("trade_pair") or "").upper()
    direction = str(dec.get("direction") or "").upper()
    try:
        conf = float(dec.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0

    by_pair = {c.trade_pair: c for c in candidates}
    if pair not in by_pair:
        print(f"gate: pair {pair!r} not in candidates")
        return None
    if direction not in ("LONG", "SHORT"):
        print(f"gate: bad direction {direction!r}")
        return None
    if conf < MIN_CONFIDENCE:
        print(f"gate: confidence {conf:.2f} < {MIN_CONFIDENCE}")
        return None

    snap = by_pair[pair]
    if snap.pre_score < MIN_PRE_SCORE:
        print(f"gate: pre_score {snap.pre_score:.2f} < {MIN_PRE_SCORE}")
        return None

    if REQUIRE_TECH_ALIGN and snap.tech_bias not in ("NONE", direction):
        print(f"gate: LLM {direction} conflicts tech_bias {snap.tech_bias}")
        return None

    # Soft news conflict: allow only with higher confidence.
    if snap.news_bias not in ("NONE", direction) and conf < MIN_CONFIDENCE + 0.1:
        print(f"gate: news conflict ({snap.news_bias}) needs higher confidence")
        return None

    return pair, direction, conf


def _post_signal_http(pair: str, direction: str, comment: str) -> dict:
    body = {"trade_pair": pair, "direction": direction, "comment": comment}
    headers = {"Content-Type": "application/json"}
    if INTAKE_TOKEN:
        headers["Authorization"] = f"Bearer {INTAKE_TOKEN}"
    req = urllib.request.Request(
        SERVE_URL, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            out = json.loads(resp.read().decode())
            out.setdefault("via", "http")
            return out
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            err = json.loads(raw)
        except json.JSONDecodeError:
            err = raw
        return {"ok": False, "error": err, "http": e.code, "via": "http"}
    except urllib.error.URLError as e:
        return {
            "ok": False,
            "error": f"miner serve unreachable at {SERVE_URL}: {e.reason}",
            "kind": "serve_down",
            "via": "http",
        }


def _post_signal_direct(pair: str, direction: str, comment: str) -> dict:
    """One-shot CLI submit via the working miner install (no REST serve needed)."""
    py = Path(MINER_PYTHON)
    miner = MINER_ROOT / "neurons" / "miner.py"
    if not py.is_file():
        return {"ok": False, "error": f"miner python missing: {py}", "via": "direct"}
    if not miner.is_file():
        return {"ok": False, "error": f"miner.py missing: {miner}", "via": "direct"}

    env = os.environ.copy()
    sn89_env = Path("/root/.sn89/env")
    if sn89_env.is_file():
        for line in sn89_env.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            if t.startswith("export "):
                t = t[7:]
            if "=" not in t:
                continue
            k, v = t.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")

    cmd = [
        str(py),
        str(miner),
        "--wallet.name",
        WALLET_NAME,
        "--wallet.hotkey",
        WALLET_HOTKEY,
        "submit",
        "--pair",
        pair,
        "--direction",
        direction,
        "--comment",
        comment,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(MINER_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "direct submit timed out (120s)", "via": "direct"}
    except OSError as e:
        return {"ok": False, "error": f"direct submit spawn failed: {e}", "via": "direct"}

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    parsed: dict | None = None
    # miner prints JSON result on stdout
    for chunk in reversed(out.split("\n\n")):
        chunk = chunk.strip()
        if chunk.startswith("{") and chunk.endswith("}"):
            try:
                parsed = json.loads(chunk)
                break
            except json.JSONDecodeError:
                continue
    if parsed is None and out.startswith("{"):
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            parsed = None

    if parsed is not None:
        parsed.setdefault("via", "direct")
        if proc.returncode != 0 and "ok" not in parsed:
            parsed["ok"] = False
        if err and not parsed.get("ok"):
            parsed.setdefault("stderr", err[-800:])
        return parsed

    # CLI limit / validation messages often go to stderr without JSON
    msg = err or out or f"exit {proc.returncode}"
    kind = None
    low = msg.lower()
    if "daily cap" in low or "quota" in low or "refused (quota)" in low:
        kind = "quota"
    elif ("min " in low and "gap" in low) or "min_spacing" in low or "refused (gap)" in low:
        kind = "gap"
    # Strip "REFUSED (quota): " prefix for cleaner UI
    clean = msg
    for prefix in ("REFUSED (quota): ", "REFUSED (gap): ", "INVALID: "):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
    return {
        "ok": False,
        "error": clean[-1200:],
        "kind": kind,
        "via": "direct",
        "exit": proc.returncode,
    }


def post_signal(pair: str, direction: str, thesis: str, dry_run: bool) -> dict:
    comment = re.sub(r"\s+", " ", (thesis or "ai_finance_bot")[:240])
    body = {"trade_pair": pair, "direction": direction, "comment": comment}
    if dry_run:
        return {"dry_run": True, **body}

    mode = SUBMIT_MODE if SUBMIT_MODE in ("auto", "http", "direct") else "auto"
    if mode == "direct":
        return _post_signal_direct(pair, direction, comment)
    if mode == "http":
        return _post_signal_http(pair, direction, comment)

    http_res = _post_signal_http(pair, direction, comment)
    if http_res.get("ok") or http_res.get("commitment"):
        return http_res
    # Fall back when serve is down; keep HTTP error if it's a real reject (4xx/5xx body).
    if http_res.get("kind") == "serve_down" or http_res.get("http") is None:
        direct = _post_signal_direct(pair, direction, comment)
        direct["http_fallback_from"] = http_res.get("error")
        return direct
    return http_res


# ── Cycle ────────────────────────────────────────────────────────────────────
def run_cycle(dry_run: bool, top_k: int) -> int:
    bands = load_bands()
    state = load_state()
    submit_ok, block_reason = can_submit(state)
    spent = spent_today_usd(state)
    n_cycles = cycles_today(state)

    print("=" * 60)
    print(f"AI finance bot  model={OPENAI_MODEL}  top_k={top_k}")
    print(f"min_conf={MIN_CONFIDENCE}  min_pre_score={MIN_PRE_SCORE}  "
          f"tech_align={REQUIRE_TECH_ALIGN}")
    print(f"budget today≈${spent:.4f} / ${DAILY_BUDGET_USD:.2f}  "
          f"cycles={n_cycles}/{MAX_CYCLES_PER_UTC_DAY}")
    print(f"submit_allowed={submit_ok}" + (f" ({block_reason})" if not submit_ok else ""))

    if spent >= DAILY_BUDGET_USD:
        print(f"→ stop: daily LLM budget ${DAILY_BUDGET_USD:.2f} reached")
        return 0
    if n_cycles >= MAX_CYCLES_PER_UTC_DAY:
        print(f"→ stop: max {MAX_CYCLES_PER_UTC_DAY} LLM cycles/UTC day reached")
        return 0

    headlines = fetch_news(36)
    print(f"news: {len(headlines)} headlines "
          f"(BULL={sum(1 for h in headlines if h.tone=='BULL')} "
          f"BEAR={sum(1 for h in headlines if h.tone=='BEAR')} "
          f"NEUT={sum(1 for h in headlines if h.tone=='NEUTRAL')})")
    for h in headlines[:8]:
        print(f"  · [{h.tone}/{h.urgency:.2f}] {h.title}")

    snaps: list[PairSnapshot] = []
    for pair, symbol in PAIRS.items():
        snap = build_snapshot(pair, symbol, bands)
        if not snap:
            continue
        attach_news(snap, headlines)
        snap.pre_score = pre_score(snap)
        snaps.append(snap)
        n_match = len(snap.matched_headlines)
        tc = snap.tech_components or {}
        print(
            f"  {pair}: tech={snap.tech_bias}/{snap.tech_score:.2f}  "
            f"news={snap.news_bias}/{snap.news_score:.2f}  "
            f"matched={n_match}  "
            f"reach={snap.reach_score:.2f}  pre={snap.pre_score:.2f}  "
            f"atr/tp={snap.atr_to_tp_ratio}"
        )
        print(
            f"      tech→ mtf={tc.get('mtf')} regime={tc.get('regime')} "
            f"style={tc.get('entry_style')} er={tc.get('er')} "
            f"ext={tc.get('extension_vs_tp')} vol={tc.get('volume_confirm')}"
        )
        for mh in snap.matched_headlines[:3]:
            print(f"      news→ {mh}")

    if not snaps:
        print("no snapshots")
        return 1

    snaps.sort(key=lambda s: s.pre_score, reverse=True)
    candidates = [s for s in snaps if s.pre_score >= MIN_PRE_SCORE][:top_k]
    if not candidates:
        # Still allow LLM on best single name if everything is weak — with NONE bias.
        candidates = snaps[:1]
        print("all pre_scores weak — sending top-1 with strong NONE preference")
    else:
        print("shortlist:", ", ".join(
            f"{c.trade_pair}({c.pre_score:.2f})" for c in candidates
        ))

    try:
        dec = call_openai(candidates, headlines, submit_ok, block_reason)
    except Exception as e:  # noqa: BLE001
        print(f"LLM error: {e}")
        return 1

    usage = dec.pop("_usage", {})
    model = dec.pop("_model", OPENAI_MODEL)
    print("LLM decision:")
    print(json.dumps(dec, indent=2))
    cost = estimate_cost_usd(usage, model) or 0.0
    if usage:
        print(
            f"tokens prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')}  "
            f"est≈${cost:.4f}  day_total≈${spent + cost:.4f}"
        )

    state.setdefault("cycles", []).append({
        "ts": time.time(),
        "ranking": [{
            "pair": s.trade_pair, "pre": s.pre_score,
            "tech": s.tech_bias, "news": s.news_bias,
        } for s in snaps],
        "shortlist": [c.trade_pair for c in candidates],
        "decision": dec,
        "usage": usage,
        "model": model,
        "est_cost_usd": cost,
    })

    picked = gate_decision(dec, candidates)
    if not picked:
        print("→ no submit")
        save_state(state)
        return 0
    if not submit_ok:
        print(f"→ blocked by miner limits: {block_reason}")
        save_state(state)
        return 0

    pair, direction, conf = picked
    print(f"→ {'DRY-RUN ' if dry_run else ''}SUBMIT {pair} {direction} conf={conf:.2f}")
    res = post_signal(pair, direction, str(dec.get("thesis") or ""), dry_run)
    print(json.dumps(res, indent=2))
    if not dry_run and (res.get("ok") or res.get("commitment")):
        state.setdefault("submits", []).append(time.time())
    save_state(state)
    return 0 if dry_run or res.get("ok") or res.get("commitment") or res.get("dry_run") else 1


DEEP_SYSTEM = """You are a senior short-horizon crypto strategist scoring Bittensor SN89 LF calls.

Protocol constraints (binding):
- Only trade_pair + LONG/SHORT matter. TP/SL are FIXED symmetric bps; ~8h crypto horizon.
- First touch of TP = WON, SL = LOST; neither = WASH (does not help emissions).
- Hit-rate matters more than activity. Be willing to say NONE.
- confidence must estimate P(TP before SL | evidence), not vibes.

YOU own the tech score:
- Read candles_15m_tail + RSI/trend/ATR fields carefully (multi-bar path, not one print).
- Produce YOUR tech_bias + tech_score (0..1).
- heuristic_tech_* is only a weak prior — you may disagree, and must explain why.

YOU own the news read:
- The payload includes all_news: the FULL headline feed (not a filtered subset).
- Cite specific headlines by paraphrase when they matter.
- Macro headlines (CPI/Fed/ETF/hack) can affect BTC/ETH even if coin name is absent.

LONG REASONING REQUIRED (do not write one-liners):
For EVERY pair write a detailed `reasoning` field of 4–8 sentences covering, in order:
  (1) price structure / MTF / RSI / momentum / chop vs trend
  (2) YOUR tech_score justification
  (3) which headlines from all_news matter and how they lean
  (4) band reachability / wash risk for the fixed TP-SL
  (5) bull case AND bear case
  (6) why final direction+confidence (or why NONE)
Also fill tech_reason, news_reason, bull_case, bear_case, blockers with real substance
(at least 1–2 full sentences each — not fragments).
market_regime should be 2–4 sentences on the cross-asset tape.
best.why should be 3–6 sentences explaining the portfolio choice across pairs.

Return STRICT JSON:
{
  "market_regime": "long paragraph",
  "scores": [
    {
      "trade_pair": "BTCUSD",
      "tech_bias": "LONG"|"SHORT"|"NONE",
      "tech_score": 0.0,
      "tech_reason": "2+ sentences",
      "news_bias": "LONG"|"SHORT"|"NONE",
      "news_reason": "2+ sentences citing headlines",
      "direction": "LONG"|"SHORT"|"NONE",
      "confidence": 0.0,
      "wash_risk": 0.0,
      "bull_case": "2+ sentences",
      "bear_case": "2+ sentences",
      "thesis": "2+ sentences",
      "blockers": "2+ sentences",
      "reasoning": "4-8 sentence deep analysis"
    }
  ],
  "best": {
    "action": "SUBMIT"|"NONE",
    "trade_pair": null,
    "direction": null,
    "confidence": 0.0,
    "why": "3-6 sentence decision memo"
  }
}
Include ALL pairs exactly once in scores.
Only set best.action=SUBMIT if confidence is truly high for an 8h 1R band trade.
"""


def run_deep_inference(dry_run: bool = True, model: str | None = None) -> int:
    """Deep per-coin bull/bear/wash analysis (diagnostic + optional submit)."""
    model = model or os.getenv("SN89_DEEP_MODEL") or OPENAI_MODEL
    bands = load_bands()
    state = load_state()
    submit_ok, block_reason = can_submit(state)
    spent = spent_today_usd(state)
    n_cycles = cycles_today(state)

    print("=" * 60)
    print(f"DEEP inference  model={model}")
    print(f"budget today≈${spent:.4f} / ${DAILY_BUDGET_USD:.2f}  "
          f"cycles={n_cycles}/{MAX_CYCLES_PER_UTC_DAY}")
    print(f"submit_allowed={submit_ok}" + (f" ({block_reason})" if not submit_ok else ""))
    if spent >= DAILY_BUDGET_USD:
        print(f"→ stop: daily LLM budget ${DAILY_BUDGET_USD:.2f} reached")
        return 0
    if n_cycles >= MAX_CYCLES_PER_UTC_DAY:
        print(f"→ stop: max {MAX_CYCLES_PER_UTC_DAY} LLM cycles/UTC day reached")
        return 0

    headlines = fetch_news(36)
    print(f"news: {len(headlines)} headlines "
          f"(BULL={sum(1 for h in headlines if h.tone=='BULL')} "
          f"BEAR={sum(1 for h in headlines if h.tone=='BEAR')} "
          f"NEUT={sum(1 for h in headlines if h.tone=='NEUTRAL')})")

    snaps: list[PairSnapshot] = []
    for pair, symbol in PAIRS.items():
        snap = build_snapshot(pair, symbol, bands)
        if not snap:
            continue
        attach_news(snap, headlines)
        snap.pre_score = pre_score(snap)
        snaps.append(snap)
        print(
            f"  {pair}: tech={snap.tech_bias}/{snap.tech_score:.2f}  "
            f"news={snap.news_bias}/{snap.news_score:.2f}  "
            f"matched={len(snap.matched_headlines)}  "
            f"reach={snap.reach_score:.2f}  pre={snap.pre_score:.2f}"
        )
        for mh in snap.matched_headlines[:2]:
            print(f"      {mh}")

    if not snaps:
        print("no snapshots")
        return 1

    user = {
        "utc_now": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "miner_can_submit_now": submit_ok,
        "min_confidence_required": MIN_CONFIDENCE,
        "all_news": headlines_for_llm(headlines, 40),
        "pairs": [s.llm_view() for s in snaps],
        "instruction": (
            "Perform DEEP inference on every pair. "
            "Compute YOUR tech_score from candles; read ALL of all_news for each pair. "
            f"Only recommend SUBMIT if confidence>={MIN_CONFIDENCE} and wash_risk is not dominant."
        ),
    }

    if not OPENAI_API_KEY:
        print("Missing OPENAI_API_KEY")
        return 2

    body = {
        "model": model,
        "temperature": 0.25,
        "max_tokens": int(os.getenv("SN89_LLM_MAX_TOKENS", "8000")),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": DEEP_SYSTEM},
            {"role": "user", "content": json.dumps(user, separators=(",", ":"))},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    if "openrouter.ai" in OPENAI_URL:
        headers["HTTP-Referer"] = "https://github.com/DeltaCompute24/InfiniteQuant-Subnet"
        headers["X-Title"] = "SN89 deep inference"

    req = urllib.request.Request(
        OPENAI_URL, data=json.dumps(body).encode(), method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"LLM error: HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
        return 1

    content = raw["choices"][0]["message"]["content"]
    dec = json.loads(content)
    usage = raw.get("usage", {})
    cost = estimate_cost_usd(usage, model) or 0.0

    print("\nMarket regime:", dec.get("market_regime"))
    print(f"\n{'PAIR':<8} {'TECH':<12} {'NEWS':<8} {'DIR':<6} {'CONF':>6} {'WASH':>6}")
    print("-" * 88)
    scores = dec.get("scores") or []
    score_rows = []
    for row in scores:
        pair = str(row.get("trade_pair") or "").upper()
        direction = str(row.get("direction") or "NONE").upper()
        try:
            conf = float(row.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        try:
            wash = float(row.get("wash_risk") or 0)
        except (TypeError, ValueError):
            wash = 0.0
        score_rows.append((pair, direction, conf, wash, row))
    for pair, direction, conf, wash, row in sorted(score_rows, key=lambda x: -x[2]):
        tb = str(row.get("tech_bias") or "?")
        try:
            ts = float(row.get("tech_score") or 0)
            tech_s = f"{tb}/{ts:.2f}"
        except (TypeError, ValueError):
            tech_s = tb
        nb = str(row.get("news_bias") or "?")
        print(f"{pair:<8} {tech_s:<12} {nb:<8} {direction:<6} {conf:6.2f} {wash:6.2f}")
        print(f"         tech: {row.get('tech_reason')}")
        print(f"         news: {row.get('news_reason')}")
        print(f"         bull: {row.get('bull_case')}")
        print(f"         bear: {row.get('bear_case')}")
        print(f"         thesis: {row.get('thesis')}")
        print(f"         blockers: {row.get('blockers')}")
        if row.get("reasoning"):
            print(f"         reasoning: {row.get('reasoning')}")

    best = dec.get("best") or {}
    print("\nBEST:")
    print(json.dumps(best, indent=2))
    if usage:
        print(
            f"\ntokens prompt={usage.get('prompt_tokens')} "
            f"completion={usage.get('completion_tokens')}  "
            f"est≈${cost:.4f}  day_total≈${spent + cost:.4f}"
        )

    state.setdefault("cycles", []).append({
        "ts": time.time(),
        "kind": "deep",
        "decision": dec,
        "usage": usage,
        "model": model,
        "est_cost_usd": cost,
    })

    # Optional submit from deep best (same hard gates as normal cycle).
    action = str(best.get("action") or "NONE").upper()
    pair = str(best.get("trade_pair") or "").upper()
    direction = str(best.get("direction") or "").upper()
    try:
        conf = float(best.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0

    by_pair = {s.trade_pair: s for s in snaps}
    fake_dec = {
        "action": action,
        "trade_pair": pair,
        "direction": direction,
        "confidence": conf,
        "thesis": str(best.get("why") or ""),
    }
    # Deep may recommend a pair not in top-k; gate against full snap set.
    picked = None
    if action == "SUBMIT" and pair in by_pair and direction in ("LONG", "SHORT"):
        if conf >= MIN_CONFIDENCE:
            if not REQUIRE_TECH_ALIGN or by_pair[pair].tech_bias in ("NONE", direction):
                if by_pair[pair].news_bias in ("NONE", direction) or conf >= MIN_CONFIDENCE + 0.1:
                    picked = (pair, direction, conf)
                else:
                    print("gate: news conflict needs higher confidence")
            else:
                print(f"gate: tech conflict {by_pair[pair].tech_bias} vs {direction}")
        else:
            print(f"gate: confidence {conf:.2f} < {MIN_CONFIDENCE}")
    else:
        print("→ no submit from deep best")

    if picked and submit_ok:
        p, d, c = picked
        print(f"→ {'DRY-RUN ' if dry_run else ''}SUBMIT {p} {d} conf={c:.2f}")
        res = post_signal(p, d, str(best.get("why") or ""), dry_run)
        print(json.dumps(res, indent=2))
        if not dry_run and (res.get("ok") or res.get("commitment")):
            state.setdefault("submits", []).append(time.time())
    elif picked and not submit_ok:
        print(f"→ blocked by miner limits: {block_reason}")

    save_state(state)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="SN89 AI finance bot (ranked LLM pipeline)")
    p.add_argument("--once", action="store_true")
    p.add_argument("--deep", action="store_true",
                   help="deep per-coin bull/bear/wash inference")
    p.add_argument("--deep-model", default=None,
                   help="override model for --deep (default SN89_DEEP_MODEL or OPENAI_MODEL)")
    p.add_argument("--live", action="store_true", help="POST to serve")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--interval", type=int, default=14400, help="default 4h ≈ 6/day")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-confidence", type=float, default=None)
    args = p.parse_args()

    global MIN_CONFIDENCE, TOP_K
    if args.min_confidence is not None:
        MIN_CONFIDENCE = args.min_confidence
    top_k = args.top_k if args.top_k is not None else TOP_K

    dry = True if args.dry_run or not args.live else False
    if dry:
        print("DRY-RUN mode (pass --live to submit)")

    if not OPENAI_API_KEY:
        print("Missing OPENAI_API_KEY — export it in your shell (do not paste into chat).")
        return 2

    if args.deep:
        return run_deep_inference(dry_run=dry, model=args.deep_model)

    if args.once:
        return run_cycle(dry_run=dry, top_k=top_k)

    while True:
        run_cycle(dry_run=dry, top_k=top_k)
        print(f"sleep {args.interval}s …")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
