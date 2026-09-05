#!/usr/bin/env python3
"""SN89 AI miner control dashboard (local).

Flow:
  Load     → prices + charts + tech (no LLM cost)
  Estimate → GPT reasoning → review + submit

Prereqs:
  1) python neurons/miner.py ... serve --port 8089
  2) bots/.env with OPENAI_API_KEY
  3) python bots/dashboard.py --port 8765
  Open http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Strip sandbox/dev proxy vars that pm2 may bake into the process env.
# Empty-string overrides in ecosystem.config.cjs are not always enough —
# SOCKS/ALL_PROXY leftovers still break partner.infinitequant.app fetches
# and make the qualification panel look empty.
for _proxy_key in (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "SOCKS_PROXY", "socks_proxy",
    "SOCKS5_PROXY", "socks5_proxy",
):
    os.environ.pop(_proxy_key, None)

BOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BOT_DIR))

import ai_finance_bot as bot  # noqa: E402
import hf_auto_bot as hfbot  # noqa: E402

# hf_auto_bot puts the repo root on sys.path; import the validator's own
# constants rather than restating the emission window and caps here.
from sn89_signals import config as sn_config  # noqa: E402

# The original (only) HF miner slot — kept as the default so old bookmarked
# tabs / a stale frontend cache without ?miner= still resolve to something.
DEFAULT_HF_TAG = "sn89-1"


def _hf_enabled_for(tag: str) -> bool:
    p = hfbot.enable_flag_path_for(tag)
    if not p.exists():
        return True
    return hfbot.load_json(p, {"enabled": True}).get("enabled", True)


def _hf_set_enabled_for(tag: str, value: bool) -> None:
    hfbot.save_json(hfbot.enable_flag_path_for(tag), {"enabled": bool(value), "at": bot.time.time()})


def _hf_pace_info(submits_today: int, daily_cap: int, now: float) -> dict:
    """Read-only mirror of hf_auto_bot.local_pace_gate's math, for display —
    shows whether today's submissions are ahead of/behind the pro-rated
    schedule the pacing gate enforces, and (if ahead) when the next slot
    opens."""
    if not daily_cap:
        return {"enabled": hfbot.PACE_ENABLED, "on_pace": True, "budgeted_by_now": None}
    if submits_today >= daily_cap:
        # The hard 30/day cap (local_rate_gate / hf.check_rate) blocks this
        # before the pace gate is ever reached, regardless of what the
        # pro-rated line says — don't show a pace ETA earlier than the
        # actual UTC-midnight reset, that would read as "resumes sooner
        # than it actually can".
        return {"enabled": hfbot.PACE_ENABLED, "on_pace": False,
                "budgeted_by_now": daily_cap, "cap_exhausted": True,
                "next_slot_eta": "00:00 UTC (daily cap reset)"}
    day_ms = 86_400_000
    now_ms = int(now * 1000)
    day_start_ms = (now_ms // day_ms) * day_ms
    day_frac = (now_ms - day_start_ms) / day_ms
    budgeted_by_now = day_frac * daily_cap + hfbot.PACE_SLACK
    on_pace = submits_today < budgeted_by_now
    info = {
        "enabled": hfbot.PACE_ENABLED,
        "on_pace": on_pace,
        "budgeted_by_now": round(min(budgeted_by_now, daily_cap), 1),
    }
    if not on_pace:
        slot_frac = max(0.0, (submits_today - hfbot.PACE_SLACK) / daily_cap)
        eta_ms = day_start_ms + slot_frac * day_ms
        info["next_slot_eta"] = bot.datetime.fromtimestamp(
            eta_ms / 1000, tz=bot.timezone.utc).strftime("%H:%M UTC")
    return info


def _snap_public(s: bot.PairSnapshot) -> dict:
    return {
        "trade_pair": s.trade_pair,
        "last": s.last,
        "chg_1h_pct": s.chg_1h_pct,
        "chg_4h_pct": s.chg_4h_pct,
        "chg_24h_pct": s.chg_24h_pct,
        "tp_sl_bps": s.tp_sl_bps,
        "atr14_15m_bps": s.atr14_15m_bps,
        "atr_to_tp_ratio": s.atr_to_tp_ratio,
        "rsi14": round(s.rsi14, 1),
        "trend": s.trend,
        "swing_pos": round(s.swing_pos, 3),
        "tech_bias": s.tech_bias,
        "tech_score": round(s.tech_score, 3),
        "tech_components": s.tech_components,
        "news_bias": s.news_bias,
        "news_score": round(s.news_score, 3),
        "matched_headlines": s.matched_headlines,
        "reach_score": round(s.reach_score, 3),
        "pre_score": round(s.pre_score, 3),
        "candles_15m_tail": s.candles_15m[-48:],
    }


def run_market(include_news: bool = True) -> dict:
    """Live prices + tech + charts — no LLM call."""
    bands = bot.load_bands()
    state = bot.load_state()
    submit_ok, block_reason = bot.can_submit(state)
    headlines: list[bot.Headline] = []
    if include_news:
        try:
            headlines = bot.fetch_news(36)
        except Exception:
            headlines = []

    snaps: list[bot.PairSnapshot] = []
    for pair, symbol in bot.PAIRS.items():
        snap = bot.build_snapshot(pair, symbol, bands)
        if not snap:
            continue
        if headlines:
            bot.attach_news(snap, headlines)
        snap.pre_score = bot.pre_score(snap)
        snaps.append(snap)

    snaps.sort(key=lambda s: s.pre_score, reverse=True)
    return {
        "ok": True,
        "utc_now": datetime.now(timezone.utc).isoformat(),
        "submit_allowed": submit_ok,
        "submit_block_reason": None if submit_ok else block_reason,
        "min_confidence": bot.MIN_CONFIDENCE,
        "headlines": [
            {"title": h.title, "tone": h.tone, "urgency": h.urgency}
            for h in headlines[:20]
        ],
        "coins": [_snap_public(s) for s in snaps],
    }


def run_estimate(model: str | None = None, deep: bool = True) -> dict:
    model = model or bot.os.getenv("SN89_DEEP_MODEL") or bot.OPENAI_MODEL
    bands = bot.load_bands()
    state = bot.load_state()
    submit_ok, block_reason = bot.can_submit(state)
    spent = bot.spent_today_usd(state)

    headlines = bot.fetch_news(36)
    snaps: list[bot.PairSnapshot] = []
    for pair, symbol in bot.PAIRS.items():
        snap = bot.build_snapshot(pair, symbol, bands)
        if not snap:
            continue
        bot.attach_news(snap, headlines)
        snap.pre_score = bot.pre_score(snap)
        snaps.append(snap)

    snaps.sort(key=lambda s: s.pre_score, reverse=True)
    all_news = bot.headlines_for_llm(headlines, 40)

    if deep:
        system = bot.DEEP_SYSTEM
        user = {
            "utc_now": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "miner_can_submit_now": submit_ok,
            "min_confidence_required": bot.MIN_CONFIDENCE,
            "all_news": all_news,
            "pairs": [s.llm_view() for s in snaps],
            "instruction": (
                "Perform DEEP inference with LONG written reasoning on every pair "
                "(reasoning field 4–8 sentences each). "
                "Compute YOUR tech_score from candles; read ALL of all_news. "
                f"Only recommend SUBMIT if confidence>={bot.MIN_CONFIDENCE} "
                "and wash_risk is not dominant. Also include best non-NONE "
                "candidate even if below threshold."
            ),
        }
    else:
        top = [s for s in snaps if s.pre_score >= bot.MIN_PRE_SCORE][: bot.TOP_K] or snaps[:1]
        system = bot.SYSTEM_PROMPT
        user = {
            "utc_now": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "miner_can_submit_now": submit_ok,
            "miner_block_reason": None if submit_ok else block_reason,
            "min_confidence_required": bot.MIN_CONFIDENCE,
            "all_news": all_news,
            "candidates": [c.llm_view() for c in top],
            "instruction": (
                "Compute YOUR tech_score; use ALL news. Choose at most ONE. "
                f"Only SUBMIT if confidence>={bot.MIN_CONFIDENCE}."
            ),
        }

    body = {
        "model": model,
        "temperature": 0.25 if deep else 0.2,
        "max_tokens": int(bot.os.getenv("SN89_LLM_MAX_TOKENS", "8000" if deep else "6000")),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, separators=(",", ":"))},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bot.OPENAI_API_KEY}",
    }
    if "openrouter.ai" in bot.OPENAI_URL:
        headers["HTTP-Referer"] = "https://github.com/DeltaCompute24/InfiniteQuant-Subnet"
        headers["X-Title"] = "SN89 dashboard estimate"

    req = urllib.request.Request(
        bot.OPENAI_URL, data=json.dumps(body).encode(), method="POST", headers=headers
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = json.loads(resp.read().decode())
    llm = json.loads(raw["choices"][0]["message"]["content"])
    usage = raw.get("usage", {})
    cost = bot.estimate_cost_usd(usage, model) or 0.0

    # Persist cycle cost tracking (estimate counts toward budget)
    state.setdefault("cycles", []).append({
        "ts": bot.time.time(),
        "kind": "dashboard_estimate",
        "decision": llm,
        "usage": usage,
        "model": model,
        "est_cost_usd": cost,
    })
    bot.save_state(state)

    # Derive suggested pick: official best, else highest non-NONE score
    best = llm.get("best") or {}
    scores = llm.get("scores") or []
    best_non_none = None
    for row in sorted(
        scores,
        key=lambda r: float(r.get("confidence") or 0),
        reverse=True,
    ):
        d = str(row.get("direction") or "NONE").upper()
        if d in ("LONG", "SHORT"):
            best_non_none = {
                "trade_pair": str(row.get("trade_pair") or "").upper(),
                "direction": d,
                "confidence": float(row.get("confidence") or 0),
                "thesis": row.get("thesis") or row.get("why") or "",
                "wash_risk": row.get("wash_risk"),
            }
            break

    suggested = None
    action = str(best.get("action") or "NONE").upper()
    if action == "SUBMIT" and best.get("trade_pair") and best.get("direction"):
        suggested = {
            "trade_pair": str(best["trade_pair"]).upper(),
            "direction": str(best["direction"]).upper(),
            "confidence": float(best.get("confidence") or 0),
            "thesis": best.get("why") or best.get("thesis") or "",
            "source": "llm_best_submit",
        }
    elif best_non_none:
        suggested = {**best_non_none, "source": "best_non_none"}

    return {
        "ok": True,
        "utc_now": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "deep": deep,
        "submit_allowed": submit_ok,
        "submit_block_reason": None if submit_ok else block_reason,
        "min_confidence": bot.MIN_CONFIDENCE,
        "budget": {
            "spent_usd": round(spent + cost, 4),
            "limit_usd": bot.DAILY_BUDGET_USD,
            "this_call_usd": round(cost, 4),
        },
        "usage": usage,
        "headlines": [
            {"title": h.title, "tone": h.tone, "urgency": h.urgency}
            for h in headlines[:20]
        ],
        "coins": [_snap_public(s) for s in snaps],
        "llm": llm,
        "suggested": suggested,
        "serve_url": bot.SERVE_URL,
    }


def _serve_probe() -> dict:
    """Cheap check that miner REST intake is up."""
    url = bot.SERVE_URL.rsplit("/submit", 1)[0] + "/submit"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return {"up": True, "http": resp.status, "url": bot.SERVE_URL}
    except urllib.error.HTTPError as e:
        # serve only implements POST /submit — 404/405 still means process is up
        return {"up": True, "http": e.code, "url": bot.SERVE_URL}
    except Exception as e:
        return {"up": False, "error": str(e), "url": bot.SERVE_URL}


MINER_API = bot.os.getenv(
    "SN89_MINER_API",
    "https://partner.infinitequant.app/api/sn89/miner",
)
# Crypto LF grade window (hours). Matches sn89_signals.config.CLASS_HORIZON_H.
CRYPTO_HORIZON_H = float(bot.os.getenv("SN89_CRYPTO_HORIZON_H", "8"))


def _fetch_iq_miner(hotkey: str) -> dict | None:
    if not hotkey:
        return None
    url = f"{MINER_API.rstrip('/')}/{hotkey}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sn89-dashboard"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
        return data if data.get("ok") else None
    except Exception as e:
        print(f"IQ miner fetch failed for {hotkey[:8]}…: {e}", flush=True)
        return None


def _spot_symbol(trade_pair: str) -> str | None:
    return bot.PAIRS.get(str(trade_pair or "").upper())


def _pair_bands(trade_pair: str, bands: dict | None = None) -> tuple[float, float]:
    """(tp_bps, sl_bps) for a pair from the local bands file."""
    bands = bands if bands is not None else bot.load_bands()
    row = bands.get(str(trade_pair or "").upper()) or {}
    tp = float(row.get("tp_bps") or 105)
    sl = float(row.get("sl_bps") or tp)
    return tp, sl


def _fetch_mark_price(trade_pair: str) -> tuple[float | None, str]:
    sym = _spot_symbol(trade_pair)
    if not sym:
        return None, "unsupported"
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={sym}"
        px = float(bot._http_json(url, timeout=8)["price"])
        return px, "binance"
    except Exception:
        try:
            bars = bot.fetch_klines(sym, "1m", 1)
            return float(bars[-1]["c"]), "binance_kline"
        except Exception:
            return None, "error"


def _fetch_entry_at_t0(trade_pair: str, t0_unix: float) -> tuple[float | None, str]:
    """Approximate graded entry: 1m open at/after commit T0 (Binance reference)."""
    sym = _spot_symbol(trade_pair)
    if not sym or not t0_unix:
        return None, "unsupported"
    t0_ms = int(t0_unix * 1000)
    try:
        url = (
            f"{bot.BINANCE_KLINES}?symbol={sym}&interval=1m"
            f"&startTime={t0_ms - 120_000}&endTime={t0_ms + 180_000}&limit=8"
        )
        rows = bot._http_json(url, timeout=10)
        entry = None
        for r in rows:
            if int(r[0]) >= t0_ms:
                entry = float(r[1])
                break
        if entry is None and rows:
            entry = float(rows[0][1])
        if entry is not None:
            return entry, "binance_1m"
    except Exception:
        pass
    return None, "error"


def _enrich_call_prices(row: dict, bands: dict | None = None) -> dict:
    """Add entry/mark/bracket fields for open calls (Binance approx; IQ grades on tick feed)."""
    status = str(row.get("status") or "")
    if status not in ("pending", "sealed"):
        return row
    pair = str(row.get("trade_pair") or "")
    direction = str(row.get("direction") or "LONG").upper()
    t0 = float(row.get("t0_unix") or 0)
    if not pair or not t0:
        return row
    tp_bps, sl_bps = _pair_bands(pair, bands)
    entry, entry_src = _fetch_entry_at_t0(pair, t0)
    mark, mark_src = _fetch_mark_price(pair)
    out = dict(row)
    out["tp_bps"] = tp_bps
    out["sl_bps"] = sl_bps
    out["entry_price"] = round(entry, 8) if entry is not None else None
    out["mark_price"] = round(mark, 8) if mark is not None else None
    out["entry_source"] = entry_src if entry is not None else None
    out["mark_source"] = mark_src if mark is not None else None
    if entry is not None:
        sign = 1 if direction == "LONG" else -1
        out["tp_price"] = round(entry * (1 + sign * tp_bps / 10_000), 8)
        out["sl_price"] = round(entry * (1 - sign * sl_bps / 10_000), 8)
    else:
        out["tp_price"] = out["sl_price"] = None
    if entry is not None and mark is not None and entry > 0:
        raw_bps = (mark - entry) / entry * 10_000
        move_bps = raw_bps if direction == "LONG" else -raw_bps
        out["move_bps"] = round(move_bps, 1)
        out["dist_tp_bps"] = round(tp_bps - move_bps, 1)
        out["dist_sl_bps"] = round(move_bps + sl_bps, 1)
    else:
        out["move_bps"] = out["dist_tp_bps"] = out["dist_sl_bps"] = None
    out["price_note"] = "Binance approx; validator grades anchored tick feed"
    return out


def _seconds_until_utc_midnight(now: float | None = None) -> int:
    now = now if now is not None else bot.time.time()
    day_end = (int(now // 86_400) + 1) * 86_400
    return max(0, int(day_end - now))


def _wilson_lb_pct(wins: int, n: int, z: float) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom) * 100


def _calibrate_wilson_z(wins: int, n: int, target_lb_pct: float | None) -> float:
    """Match IQ API confidence_lb when possible; else ~80% CI default."""
    if n <= 0 or target_lb_pct is None:
        return 1.28
    target = float(target_lb_pct)
    best_z, best_err = 1.28, 999.0
    for i in range(80, 220):
        z = i / 100.0
        err = abs(_wilson_lb_pct(wins, n, z) - target)
        if err < best_err:
            best_z, best_err = z, err
        if err < 0.05:
            return z
    return best_z


def _gate_pass(
    wins: int,
    losses: int,
    *,
    min_decisive: int,
    min_hit_pct: float,
    lb_floor_pct: float,
    z: float,
) -> dict:
    n = wins + losses
    hit_pct = wins / n * 100 if n else 0.0
    lb_pct = _wilson_lb_pct(wins, n, z) if n else 0.0
    decisive_ok = n >= min_decisive
    hit_ok = hit_pct >= min_hit_pct
    lb_ok = lb_pct >= lb_floor_pct
    return {
        "wins": wins,
        "losses": losses,
        "decisive": n,
        "hit_pct": round(hit_pct, 1),
        "lb_pct": round(lb_pct, 1),
        "decisive_ok": decisive_ok,
        "hit_ok": hit_ok,
        "lb_ok": lb_ok,
        "qualifies": decisive_ok and hit_ok and lb_ok,
    }


# ── trailing-window earning view ─────────────────────────────────────────────
# Everything below is measured BACKWARD FROM `now`, not from a calendar or UTC
# boundary: a win leaves the window exactly EMISSION_DECAY_S after its own t0.
# The quota timers elsewhere on the panel do reset at UTC midnight, so the UI
# says "Past N days" rather than "this week" to keep the two apart.
# The scoreline above the panel is LIFETIME, but emission is sized by a tally
# that decays to zero over exactly one week (config.EMISSION_DECAY_S on LF,
# hf.HF_EMISSION_DECAY_S on HF — both 7d since 2026-07-31). A miner whose last
# win is eight days old reads healthy on the lifetime line while contributing
# nothing to the vector, so the panel needs the window the validator pays on.
#
# This is deliberately NOT scoring.decayed_qwin_tally, and must not be labelled
# as it. That function weights every win by the tier AND wash-efficiency it held
# at its own t0, and neither input exists in the call feed. What is honestly
# reproducible here is the unweighted SHAPE: each win counts 1.0 and decays
# linearly across the same window under the same WIN_CAP. Same curve and same
# horizon, minus the per-win multiplier.
#
# The count is also only a numerator — compute_weights normalizes across the
# field, so this rises and falls with the rest of the field's activity too.
def _weekly_rollup(
    calls: list[dict],
    now: float,
    *,
    decay_s: int,
    win_cap: int,
    min_decisive: int,
    lb_floor_pct: float,
    z: float = 1.2816,
    eligible: bool = True,
    lifetime: tuple[int, int] | None = None,
) -> dict:
    """Win counts over the emission window plus an unweighted decay proxy.

    `calls` is any per-call ledger carrying `t0_unix` and `status` (the LF
    /api/status feed and the HF results ledger both qualify). Wins are split
    into raw and gate-qualified, the latter re-deriving the qualify gate AS OF
    each win the way scoring.qualified_wins does.

    `lifetime` is IQ's (won, lost) totals. The call feed is PAGED — LF returns
    ~25 rows and HF is capped — so it routinely covers days where the gate reads
    60. Reconstructing from the feed alone restarts the miner's record from zero
    and reports a long-qualified miner as unqualified, so whatever IQ counts but
    the feed does not show is folded back in as a bulk seed on the trailing
    window. Exact when the career fits inside the reputation window; an
    overcount for a career older than it, which is why anything seeded is
    flagged approximate.
    """
    decisive: list[tuple[float, bool]] = []
    washed = 0
    for c in calls:
        t0 = float(c.get("t0_unix") or 0)
        st = str(c.get("status") or "").lower()
        if not t0:
            continue
        if st in ("won", "lost"):
            decisive.append((t0, st == "won"))
        elif st == "washed" and 0.0 <= now - t0 < decay_s:
            washed += 1
    decisive.sort(key=lambda d: d[0])

    def _live(t0: float) -> bool:
        return 0.0 <= now - t0 < decay_s

    live_wins = [t0 for t0, won in decisive if won and _live(t0)]
    lost = sum(1 for t0, won in decisive if not won and _live(t0))

    # Qualification AS OF each win, mirroring scoring.qualified_wins: the
    # trailing HIT_RATE_WINDOW_S ending at that win (the win itself included),
    # capped at the most recent hit_rate_window_trades_as_of(t0) outcomes. The
    # live gate is CONFIDENCE_SCORING — sample floor plus Wilson LB, with no
    # raw-hit term — so only those two conditions are applied.
    # Outcomes IQ has graded that this page of the feed never showed us.
    seed_wins = seed_decisive = 0
    if lifetime:
        lt_won, lt_lost = lifetime
        seed_wins = max(0, int(lt_won) - sum(1 for _, w in decisive if w))
        seed_decisive = max(0, (int(lt_won) + int(lt_lost)) - len(decisive))
        seed_wins = min(seed_wins, seed_decisive)

    qualified: list[float] = []
    for i, (t0, won) in enumerate(decisive):
        if not won or not _live(t0):
            continue
        cut = t0 - sn_config.HIT_RATE_WINDOW_S
        cap = sn_config.hit_rate_window_trades_as_of(t0)
        window = [d for d in decisive[:i + 1] if d[0] >= cut][-cap:]
        rep_wins = sum(1 for _, w in window if w)
        rep_decisive = len(window)
        # The unseen remainder is older than everything in the feed, so it fills
        # what the trade cap leaves over after the rows we can actually see.
        if seed_decisive:
            used = min(seed_decisive, max(0, cap - rep_decisive))
            if used:
                rep_wins += int(round(seed_wins * used / seed_decisive))
                rep_decisive += used
        if rep_decisive >= min_decisive and _wilson_lb_pct(rep_wins, rep_decisive, z) >= lb_floor_pct:
            qualified.append(t0)

    def _decay_sum(stamps: list[float]) -> float:
        # Most recent first, then WIN_CAP-truncated, exactly as the tally does.
        newest = sorted(stamps, reverse=True)[:win_cap]
        return round(sum(1.0 - (now - t0) / decay_s for t0 in newest), 3)

    # The as-of gate reads a trailing reputation window. When the feed does not
    # reach that far back before the oldest scored win — a truncated ledger, or
    # simply a short career, which look identical from here — the sample is
    # short and qualified wins can be undercounted.
    approx = bool(live_wins) and (
        bool(seed_decisive)
        or decisive[0][0] > min(live_wins) - sn_config.HIT_RATE_WINDOW_S
    )

    return {
        "window_days": round(decay_s / 86_400.0, 2),
        "won": len(live_wins),
        "lost": lost,
        "washed": washed,
        "qualified_won": len(qualified) if eligible else 0,
        "decay_sum": _decay_sum(live_wins),
        "decay_sum_qualified": _decay_sum(qualified) if eligible else 0.0,
        "win_cap": win_cap,
        "cap_binding": len(live_wins) > win_cap,
        "newest_win_age_h": round((now - max(live_wins)) / 3600.0, 1) if live_wins else None,
        "qualified_approx": approx,
        "eligible": bool(eligible),
    }


def _build_qualify_path(
    miner: dict,
    config: dict,
) -> dict | None:
    if not miner:
        return None
    min_decisive = int(config.get("qualify_min_decisive") or 8)
    min_hit_pct = float(config.get("qualify_min_hit_pct") or 55.0)
    lb_floor_pct = float(config.get("qualify_lb_floor_pct") or 50.0)
    wins = int(miner.get("won") or 0)
    losses = int(miner.get("lost") or 0)
    n = wins + losses
    api_lb = miner.get("confidence_lb_pct")
    z = _calibrate_wilson_z(wins, n, api_lb)
    current = _gate_pass(
        wins,
        losses,
        min_decisive=min_decisive,
        min_hit_pct=min_hit_pct,
        lb_floor_pct=lb_floor_pct,
        z=z,
    )
    if api_lb is not None:
        current["lb_pct"] = round(float(api_lb), 1)
        current["lb_ok"] = current["lb_pct"] >= lb_floor_pct
        current["qualifies"] = (
            current["decisive_ok"] and current["hit_ok"] and current["lb_ok"]
        )

    scenarios: list[dict] = []
    for label, outcome in (("Next win", "W"), ("Next loss", "L")):
        nw, nl = (wins + 1, losses) if outcome == "W" else (wins, losses + 1)
        row = _gate_pass(
            nw,
            nl,
            min_decisive=min_decisive,
            min_hit_pct=min_hit_pct,
            lb_floor_pct=lb_floor_pct,
            z=z,
        )
        row["label"] = label
        row["path"] = outcome
        scenarios.append(row)

    for streak in range(2, 5):
        row = _gate_pass(
            wins + streak,
            losses,
            min_decisive=min_decisive,
            min_hit_pct=min_hit_pct,
            lb_floor_pct=lb_floor_pct,
            z=z,
        )
        row["label"] = f"{streak}-win streak"
        row["path"] = "W" * streak
        scenarios.append(row)

    min_trades_to_qualify: int | None = None
    min_wins_needed: int | None = None
    seen = {(wins, losses)}
    queue: list[tuple[int, int, int, int]] = [(wins, losses, 0, 0)]
    while queue:
        cw, cl, depth, extra_w = queue.pop(0)
        if (cw, cl) in seen and depth > 0:
            continue
        seen.add((cw, cl))
        row = _gate_pass(
            cw,
            cl,
            min_decisive=min_decisive,
            min_hit_pct=min_hit_pct,
            lb_floor_pct=lb_floor_pct,
            z=z,
        )
        if row["qualifies"]:
            min_trades_to_qualify = depth
            min_wins_needed = extra_w
            break
        if depth >= 8:
            continue
        for outcome in ("W", "L"):
            nw, nl = (cw + 1, cl) if outcome == "W" else (cw, cl + 1)
            if (nw, nl) not in seen:
                queue.append((nw, nl, depth + 1, extra_w + (1 if outcome == "W" else 0)))

    return {
        "meets_gate": bool(miner.get("meets_gate")),
        "thresholds": {
            "min_decisive": min_decisive,
            "min_hit_pct": min_hit_pct,
            "lb_floor_pct": lb_floor_pct,
        },
        "current": current,
        "gaps": {
            "decisives_left": max(0, min_decisive - n),
            "lb_gap_pct": round(max(0.0, lb_floor_pct - current["lb_pct"]), 1),
            "decisive_ok": current["decisive_ok"],
            "hit_ok": current["hit_ok"],
            "lb_ok": current["lb_ok"],
        },
        "min_trades_to_qualify": min_trades_to_qualify,
        "min_wins_needed": min_wins_needed,
        "scenarios": scenarios,
    }


def run_status() -> dict:
    """Live desk status: quotas, timers, and IQ submission history."""
    now = bot.time.time()
    state = bot.load_state()
    submit_ok, block_reason = bot.can_submit(state)
    hotkey = bot._resolve_hotkey_ss58()
    ts = bot._dedupe_submit_ts(
        [float(t) for t in state.get("submits", [])] + bot.miner_submit_timestamps(hotkey)
    )
    day = int(now // 86_400)
    today = [t for t in ts if int(t // 86_400) == day]
    last_ts = max(ts) if ts else None
    gap_left = 0
    if last_ts is not None and now - last_ts < bot.MIN_GAP_S:
        gap_left = int(bot.MIN_GAP_S - (now - last_ts))

    iq = _fetch_iq_miner(hotkey) or {}
    miner = iq.get("miner") or {}
    iq_config = iq.get("config") or {}
    qualify_path = _build_qualify_path(miner, iq_config)
    bands = bot.load_bands()
    calls_raw = iq.get("calls") or []
    calls: list[dict] = []
    pending_timers: list[dict] = []
    open_call: dict | None = None
    for c in sorted(calls_raw, key=lambda x: float(x.get("t0_unix") or 0), reverse=True):
        t0 = float(c.get("t0_unix") or 0)
        status = str(c.get("status") or "")
        horizon_h = CRYPTO_HORIZON_H
        horizon_end = t0 + horizon_h * 3600 if t0 else None
        left = int(horizon_end - now) if horizon_end and status in ("pending", "sealed") else None
        row = {
            "t0_unix": t0 or None,
            "t0_utc": datetime.fromtimestamp(t0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            if t0 else None,
            "trade_pair": c.get("trade_pair"),
            "direction": c.get("direction"),
            "status": status,
            "outcome_bps": c.get("outcome_bps"),
            "exit_reason": c.get("exit_reason"),
            "void_reason": c.get("void_reason"),
            "reveal_unix": c.get("reveal_unix"),
            "horizon_h": horizon_h,
            "horizon_end_unix": horizon_end,
            "horizon_end_utc": datetime.fromtimestamp(horizon_end, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            )
            if horizon_end else None,
            "horizon_left_s": max(0, left) if left is not None and left > 0 else (0 if left is not None else None),
        }
        if status in ("pending", "sealed"):
            row = _enrich_call_prices(row, bands)
            if open_call is None:
                open_call = row
        calls.append(row)
        if status in ("pending", "sealed") and left is not None and left > 0:
            pending_timers.append({
                "trade_pair": row["trade_pair"] or "?",
                "direction": row["direction"] or "?",
                "status": status,
                "t0_unix": t0,
                "horizon_end_unix": horizon_end,
                "left_s": max(0, left),
                "entry_price": row.get("entry_price"),
                "mark_price": row.get("mark_price"),
                "move_bps": row.get("move_bps"),
                "dist_tp_bps": row.get("dist_tp_bps"),
            })

    week = _weekly_rollup(
        calls,
        now,
        decay_s=sn_config.EMISSION_DECAY_S,
        win_cap=sn_config.WIN_CAP,
        min_decisive=int(iq_config.get("qualify_min_decisive") or 8),
        lb_floor_pct=float(iq_config.get("qualify_lb_floor_pct") or 50.0),
        lifetime=(int(miner.get("won") or 0), int(miner.get("lost") or 0)) if miner else None,
    )

    return {
        "ok": True,
        "utc_now": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "now_unix": now,
        "wallet": f"{bot.WALLET_NAME}/{bot.WALLET_HOTKEY}",
        "hotkey": hotkey,
        "serve": _serve_probe(),
        "submit_allowed": submit_ok,
        "submit_block_reason": None if submit_ok else block_reason,
        "quota": {
            "today_used": len(today),
            "max_per_day": bot.MAX_PER_UTC_DAY,
            "iq_today_used": miner.get("pace", {}).get("today_used") if miner else None,
            "min_gap_s": bot.MIN_GAP_S,
            "last_submit_unix": last_ts,
            "last_submit_utc": datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            if last_ts else None,
        },
        "timers": {
            "gap_left_s": gap_left,
            "day_reset_s": _seconds_until_utc_midnight(now),
            "pending": pending_timers,
        },
        "score": {
            "won": miner.get("won"),
            "lost": miner.get("lost"),
            "washed": miner.get("washed"),
            "void": miner.get("void"),
            "pending": miner.get("pending"),
            "sealed": miner.get("sealed"),
            "hit_pct": miner.get("rep_hit_pct"),
            "lifetime_decisive": miner.get("lifetime_decisive"),
            "confidence_lb_pct": miner.get("confidence_lb_pct"),
            "meets_gate": miner.get("meets_gate"),
            "status": miner.get("status"),
        } if miner else None,
        "qualify_path": qualify_path,
        # Withheld when IQ is unreachable: an empty call feed and a genuinely
        # winless week are the same zeros, and only one of them is true.
        "week": week if miner else None,
        "open_call": open_call,
        "calls": calls,
        "iq_ok": bool(miner),
    }


# ── HF auto-bot panel ─────────────────────────────────────────────────────
# Read-only view of a SEPARATE process (bots/hf_auto_bot.py, normally run under
# PM2 as sn89-hf-auto): this dashboard does not hold the HF tick buffers or
# submit HF calls itself, it only reads the state file that bot writes every
# cycle and flips the enable flag file it polls. Loosely coupled on purpose —
# the automation keeps running even if the dashboard is restarted or down.
def _hf_binance_mark(pair: str) -> float | None:
    """Best-effort live reference price for pairs Binance also lists (crypto).
    Non-Binance HF pairs (XAUUSD, AUDUSD, HYPEUSD, ...) fall back to the bot's
    own last-seen tick price from its state file — there is no cheap public
    spot source for those here."""
    try:
        px, _src = _fetch_mark_price(pair)
        return px
    except Exception:
        return None


def _enrich_hf_ledger_calls(calls: list[dict]) -> list[dict]:
    """Add live move_bps for open HF ledger rows; resolved rows use outcome_bps."""
    marks: dict[str, float | None] = {}
    out: list[dict] = []
    for row in calls:
        r = dict(row)
        st = str(r.get("status") or "").lower()
        pair = str(r.get("trade_pair") or "")
        direction = str(r.get("direction") or "LONG").upper()
        if st in ("open", "pending") and pair:
            if pair not in marks:
                marks[pair] = _hf_binance_mark(pair)
            entry = r.get("entry_price")
            if entry is None and r.get("t0_unix"):
                fetched, _src = _fetch_entry_at_t0(pair, float(r["t0_unix"]))
                if fetched is not None:
                    entry = fetched
                    r["entry_price"] = round(entry, 8)
            mark = marks.get(pair)
            if mark is not None:
                r["mark_price"] = round(mark, 8)
            if entry is not None and mark is not None and entry > 0:
                sign = 1 if direction == "LONG" else -1
                r["move_bps"] = round((mark - entry) / entry * 10_000 * sign, 1)
            else:
                r["move_bps"] = None
        elif r.get("outcome_bps") is not None:
            r["move_bps"] = r["outcome_bps"]
        else:
            r["move_bps"] = None
        out.append(r)
    return out


def run_hf_status(tag: str = DEFAULT_HF_TAG) -> dict:
    state = hfbot.load_json(hfbot.state_path_for(tag), {})
    hotkey = state.get("hotkey") or bot._resolve_hotkey_ss58()
    now = bot.time.time()
    board = hfbot.hf.hf_bands_as_of(now) or {}
    if hfbot.PAIR_ALLOWLIST:
        # Mirror the bot's own restriction (SN89_HF_AUTO_PAIRS) so the panel
        # doesn't show pairs it will never actually evaluate.
        board = {p: row for p, row in board.items() if p in hfbot.PAIR_ALLOWLIST}

    price_by_pair = {c.get("pair"): c.get("price") for c in (state.get("candidates") or [])}

    # open_calls comes straight from the bot's own receipts file every cycle
    # (hf_auto_bot._open_calls_by_pair), NOT from `decisions` — that log is
    # capped at 100 entries and gets deduped/evicted independently of how
    # long a position stays open, so it's not a reliable place to read
    # "is this still open" from.
    open_positions = []
    for c in state.get("open_calls") or []:
        row = dict(c)
        end = (row["t0_ms"] / 1000.0) + float(row.get("horizon_s") or 0)
        left = end - now
        row["horizon_end_unix"] = end
        row["horizon_left_s"] = max(0, int(left))
        row["entry_ref"] = row.get("entry")
        mark = _hf_binance_mark(row["pair"]) or price_by_pair.get(row["pair"])
        row["mark_price"] = round(mark, 8) if mark is not None else None
        entry = row.get("entry")
        if entry and mark and entry > 0:
            sign = 1 if row.get("direction") == "LONG" else -1
            move_bps = (mark - entry) / entry * 10_000 * sign
            row["move_bps"] = round(move_bps, 1)
        else:
            row["move_bps"] = None
        if left > 0:
            open_positions.append(row)

    history = list(state.get("decisions") or [])

    return {
        "ok": True,
        "utc_now": bot.datetime.now(bot.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tag": tag,
        "hotkey": hotkey,
        "enabled": _hf_enabled_for(tag),
        "dry_run": state.get("dry_run"),
        # Staleness threshold must track EVAL_INTERVAL_S — state is only
        # rewritten once per cycle, so a fixed 15s threshold (fine when the
        # loop ran every 1.5s) falsely reported "not running" for most of
        # each cycle once the cadence was slowed to 300s (2026-08-31). 2x +
        # 30s slack tolerates one slow cycle (e.g. a laggy SSE reconnect)
        # without flapping the badge.
        "running": bool(state) and (now - float(state.get("updated_at", 0))
                                     < hfbot.EVAL_INTERVAL_S * 2 + 30),
        "state_age_s": (now - float(state["updated_at"])) if state.get("updated_at") else None,
        "board": {
            pair: {"tp_bps": row[0], "sl_bps": row[1], "horizon_s": row[2], "asset_class": row[3]}
            for pair, row in board.items()
        },
        "submits_today": state.get("submits_today", 0),
        "daily_cap": hfbot.hf.hf_rules_as_of(now)[0],
        "pace": _hf_pace_info(state.get("submits_today", 0), hfbot.hf.hf_rules_as_of(now)[0], now),
        # Written by hf_auto_bot._write_state — which pair, if any, ROTATE_DAILY
        # currently allows this instance to act on (others still score, for
        # visibility, but can never be the one that submits).
        "rotation": state.get("rotation") or {"enabled": False},
        # check_rate's daily cap resets on a UTC calendar-day boundary (day =
        # t_ms // 86_400_000), not a rolling 24h window — this is that
        # boundary in seconds-from-now, so the dashboard can show "resets in
        # Xh Ym" instead of a bare "00:00 UTC" that doesn't say how soon that
        # actually is.
        "seconds_to_daily_reset": int(86_400 - (now % 86_400)),
        "candidates": state.get("candidates") or [],
        "open_positions": open_positions,
        "history": history[:100],
    }


def run_hf_results(tag: str = DEFAULT_HF_TAG) -> dict:
    """Won/lost/wash + eligibility, straight from the validator-side status API
    (_fetch_iq_miner) — grading and the eligibility gate (>=50 accepted across
    >=8 distinct UTC days) both happen server-side, this bot has no local view
    of either."""
    state = hfbot.load_json(hfbot.state_path_for(tag), {})
    hotkey = state.get("hotkey") or bot._resolve_hotkey_ss58()
    data = _fetch_iq_miner(hotkey)
    hf = (data or {}).get("hf")
    if not hf:
        return {"ok": False, "hotkey": hotkey, "tag": tag,
                "error": "IQ API unavailable or no HF submissions yet"}

    gate = hf.get("gate") or {}
    diversity = gate.get("diversity") or {}
    pace = hf.get("pace") or {}
    # Roll the week up over the FULL ledger, not the 300-row display slice, so
    # the as-of qualify gate reads as much trailing history as IQ will give us.
    hf_calls = sorted(hf.get("calls") or [], key=lambda c: c.get("t0_unix") or 0, reverse=True)
    week = _weekly_rollup(
        hf_calls,
        bot.time.time(),
        decay_s=hfbot.hf.HF_EMISSION_DECAY_S,
        win_cap=hfbot.hf.HF_WIN_CAP,
        min_decisive=hfbot.hf.HF_QUALIFY_MIN_DECISIVE,
        lb_floor_pct=hfbot.hf.HF_QUALIFY_LB_FLOOR * 100.0,
        # Nothing earns before the HF eligibility gate opens, however the
        # confidence gate reads.
        eligible=bool(hf.get("eligible")),
        lifetime=(int(hf.get("won") or 0), int(hf.get("lost") or 0)),
    )
    return {
        "ok": True,
        "tag": tag,
        "hotkey": hotkey,
        "week": week,
        "snapshot_at": hf.get("snapshot_at"),
        "status": hf.get("status"),
        "qualified": bool(hf.get("qualified")),
        "eligible": bool(hf.get("eligible")),
        "won": int(hf.get("won") or 0),
        "lost": int(hf.get("lost") or 0),
        "washed": int(hf.get("washed") or 0),
        "pending": int(hf.get("pending") or 0),
        "hit_rate_pct": hf.get("hit_rate_pct"),
        # IQ's conf_hit_pct is the SHRUNK hit-rate (tier metric), NOT Wilson LB.
        # We still pass it through for display, and compute the real qualify
        # gate number ourselves from won/lost with the same z the validator uses.
        "conf_hit_pct": hf.get("conf_hit_pct"),
        "wilson_lb_pct": round(
            _wilson_lb_pct(int(hf.get("won") or 0),
                           int(hf.get("won") or 0) + int(hf.get("lost") or 0),
                           1.2816),
            1,
        ),
        "wilson_lb_ok": (
            (int(hf.get("won") or 0) + int(hf.get("lost") or 0)) >= 8
            and _wilson_lb_pct(int(hf.get("won") or 0),
                               int(hf.get("won") or 0) + int(hf.get("lost") or 0),
                               1.2816) >= 50.0
        ),
        "avg_r": hf.get("avg_r"),
        "avg_mfe_bps": hf.get("avg_mfe_bps"),
        "avg_mae_bps": hf.get("avg_mae_bps"),
        "emission_weight": hf.get("emission_weight"),
        "gate": {
            "submissions": gate.get("submissions"),
            "submissions_required": gate.get("submissions_required"),
            "submissions_remaining": gate.get("submissions_remaining"),
            "trading_days": gate.get("trading_days"),
            "trading_days_required": gate.get("trading_days_required"),
            "trading_days_remaining": gate.get("trading_days_remaining"),
            "last_trading_day": gate.get("last_trading_day"),
            "diversity_ok": gate.get("diversity_ok"),
            "diversity": {
                "pairs": diversity.get("pairs"),
                "long": diversity.get("long"),
                "short": diversity.get("short"),
                "minority": diversity.get("minority"),
                "share": diversity.get("share"),
                "floor": diversity.get("floor"),
                "applies": diversity.get("applies"),
                "ok": diversity.get("ok"),
                "by_pair": diversity.get("by_pair") or {},
            },
        },
        "assets": hf.get("assets") or [],
        "streak": pace.get("streak"),
        # Per-submission ledger straight from the validator's call list — the
        # ONLY authoritative per-call view (won/lost/open/refused), unlike
        # /api/hf/status's open_positions which is this bot's own local
        # receipts + live mark price. Newest first; capped defensively since
        # the eligibility gate alone already bounds daily volume at 30/day.
        "calls": _enrich_hf_ledger_calls(hf_calls[:300]),
    }


def run_hf_toggle(enabled_value: bool, tag: str = DEFAULT_HF_TAG) -> dict:
    _hf_set_enabled_for(tag, enabled_value)
    return {"ok": True, "tag": tag, "enabled": enabled_value}


def run_hf_manual_submit(pair: str, direction: str, tag: str = DEFAULT_HF_TAG) -> dict:
    """Queue a one-shot manual submit for the RUNNING hf_auto_bot instance
    identified by `tag` (see hf_auto_bot.manual_request_path_for). We don't
    call submit_hf() ourselves here — the dashboard process has no live tick
    buffer or up-to-date open-position/rate state for that hotkey, and a
    second code path racing the bot's own submit_log/receipts is exactly the
    kind of race the bot's single-decision-loop design exists to avoid.
    Dropping a file for the bot to pick up on its next tick keeps this a
    real click through the SAME gates (rate/pair-open/cross-lock) an
    automatic submission would go through, just without the pacing throttle
    or scoring/streak requirements."""
    pair = str(pair or "").upper().strip()
    direction = str(direction or "").upper().strip()
    if direction not in ("LONG", "SHORT"):
        return {"ok": False, "error": "direction must be LONG or SHORT"}
    now = bot.time.time()
    board = hfbot.hf.hf_bands_as_of(now) or {}
    if hfbot.PAIR_ALLOWLIST:
        board = {p: row for p, row in board.items() if p in hfbot.PAIR_ALLOWLIST}
    if pair not in board:
        return {"ok": False, "error": f"pair not on the live HF board right now: {pair}"}
    state = hfbot.load_json(hfbot.state_path_for(tag), {})
    running = bool(state) and (now - float(state.get("updated_at", 0))
                                < hfbot.EVAL_INTERVAL_S * 2 + 30)
    if not running:
        return {"ok": False, "error": f"miner '{tag}' does not look like it's running "
                                       "right now — start it in PM2 first"}
    hfbot.save_json(hfbot.manual_request_path_for(tag),
                    {"pair": pair, "direction": direction, "requested_at": now})
    return {
        "ok": True,
        "tag": tag,
        "pair": pair,
        "direction": direction,
        "dry_run": state.get("dry_run"),
        "note": "queued — the running bot picks this up within a couple of seconds and "
                "runs it through the same rate/pair-open/cross-lock gates as an automatic "
                "submission (pacing is skipped for manual clicks). Watch the ledger below.",
    }


def run_hf_manual_submit_all(pair: str, direction: str) -> dict:
    """Fan a manual submit out to EVERY discovered HF miner instance (per
    user request 2026-08-31: "if I submit manually, submit both"), not just
    whichever tab happens to be selected. Each miner still runs the request
    through its OWN independent gates (rate/pair-open/cross-lock are scoped
    per-hotkey, see run_hf_manual_submit) — one miner already holding an
    open position on this pair blocking IT does not block the other."""
    miners = (run_hf_miners().get("miners") or [])
    if not miners:
        return {"ok": False, "error": "no HF miners found"}
    results = {}
    for m in miners:
        results[m["tag"]] = run_hf_manual_submit(pair, direction, m["tag"])
    return {
        "ok": any(r.get("ok") for r in results.values()),
        "pair": str(pair or "").upper().strip(),
        "direction": str(direction or "").upper().strip(),
        "results": results,
    }


def run_hf_miners() -> dict:
    """Every HF miner instance discovered on this box — one state file per
    hf_auto_bot.py process (see SN89_HF_AUTO_TAG). Drives the dashboard's
    miner switcher; a fresh box with only sn89-hf-auto running still returns
    exactly one entry."""
    now = bot.time.time()
    miners = []
    for p in sorted(hfbot.SN89_DIR.glob("hf_auto_state_*.json")):
        tag = p.stem[len("hf_auto_state_"):]
        state = hfbot.load_json(p, {})
        updated_at = state.get("updated_at")
        miners.append({
            "tag": tag,
            "hotkey": state.get("hotkey"),
            "pairs": state.get("pairs") or [],
            "enabled": _hf_enabled_for(tag),
            "dry_run": state.get("dry_run"),
            "running": bool(state) and bool(updated_at)
                       and (now - float(updated_at) < hfbot.EVAL_INTERVAL_S * 2 + 30),
            "submits_today": state.get("submits_today", 0),
        })
    return {"ok": True, "miners": miners, "default": DEFAULT_HF_TAG}


def run_submit(pair: str, direction: str, thesis: str = "") -> dict:
    pair, direction = pair.upper(), direction.upper()
    if pair not in bot.PAIRS:
        return {"ok": False, "error": f"pair not allowed: {pair}"}
    if direction not in ("LONG", "SHORT"):
        return {"ok": False, "error": "direction must be LONG or SHORT"}

    state = bot.load_state()
    ok, reason = bot.can_submit(state)
    if not ok:
        return {"ok": False, "error": reason, "kind": "limit"}

    res = bot.post_signal(pair, direction, thesis or "dashboard submit", dry_run=False)
    committed = bool(res.get("ok") is True or (res.get("commitment") and res.get("ok") is not False))
    err = res.get("error")
    if isinstance(err, dict):
        err_msg = err.get("error") or json.dumps(err)[:500]
        kind = err.get("kind")
    else:
        err_msg = err
        kind = res.get("kind")
    if committed:
        state.setdefault("submits", []).append(bot.time.time())
        bot.save_state(state)
    return {
        "ok": committed,
        "pair": pair,
        "direction": direction,
        "serve_url": bot.SERVE_URL,
        "submit_mode": bot.SUBMIT_MODE,
        "via": res.get("via"),
        "error": None if committed else (err_msg or "submit failed"),
        "kind": kind,
        "result": res,
    }


HTML = r"""<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
<title>SN89 Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {
  theme: {
    extend: {
      fontFamily: {
        sans: ['IBM Plex Sans', 'system-ui', 'sans-serif'],
        mono: ['IBM Plex Mono', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 40px -10px rgba(16,185,129,0.35)',
        panel: '0 25px 50px -12px rgba(0,0,0,0.55)',
      },
    },
  },
};
</script>
<style type="text/tailwindcss">
  @layer components {
    .desk-panel {
      @apply rounded-2xl border border-zinc-800/90 bg-zinc-900/45 backdrop-blur-md shadow-panel;
    }
    .desk-panel-head {
      @apply text-[11px] font-bold uppercase tracking-[0.14em] text-zinc-500 mb-4 flex items-center gap-2;
    }
    .desk-panel-head::before {
      content: '';
      @apply w-1 h-3.5 rounded-full bg-gradient-to-b from-emerald-400 to-cyan-500;
    }
    .desk-input {
      @apply rounded-xl border border-zinc-700/90 bg-zinc-950/70 px-3.5 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500/50 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 transition;
    }
    .desk-select {
      @apply desk-input pr-9 appearance-none cursor-pointer;
    }
    .desk-table { @apply w-full text-sm border-collapse; }
    .desk-th { @apply px-3 py-2.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-500 border-b border-zinc-800 bg-zinc-950/40 text-left; }
    .desk-td { @apply px-3 py-2.5 border-b border-zinc-800/50 align-top; }
    .desk-chip { @apply inline-flex items-center gap-1.5 rounded-full border border-zinc-800 bg-zinc-900/90 px-3 py-1 text-xs font-medium text-zinc-300; }
    .desk-timer-card { @apply desk-panel p-5 relative overflow-hidden; }
    .desk-timer-card::after {
      content: '';
      @apply pointer-events-none absolute -right-6 -top-6 h-24 w-24 rounded-full bg-emerald-500/5 blur-2xl;
    }
    .btn-primary { @apply inline-flex items-center justify-center rounded-xl bg-gradient-to-b from-white to-zinc-200 px-5 py-2.5 text-sm font-bold text-zinc-950 shadow-lg hover:from-zinc-50 transition active:scale-[0.98] disabled:opacity-40; }
    .btn-secondary { @apply inline-flex items-center justify-center rounded-xl border border-zinc-700 bg-zinc-900/80 px-4 py-2.5 text-sm font-semibold text-zinc-200 hover:border-zinc-500 hover:bg-zinc-800 transition disabled:opacity-40; }
    .btn-good { @apply inline-flex items-center justify-center rounded-xl bg-gradient-to-b from-emerald-400 to-emerald-600 px-5 py-2.5 text-sm font-bold text-emerald-950 shadow-lg shadow-emerald-900/30 hover:from-emerald-300 transition disabled:opacity-40; }
    .btn-sm-ghost { @apply inline-flex items-center rounded-lg border border-zinc-700 bg-zinc-900/60 px-2.5 py-1 text-[11px] font-bold text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition disabled:opacity-40; }
    .btn-sm-long { @apply btn-sm-ghost hover:border-emerald-500/40 hover:text-emerald-300; }
    .btn-sm-short { @apply btn-sm-ghost hover:border-rose-500/40 hover:text-rose-300; }
    .btn-sm-good { @apply inline-flex items-center rounded-lg bg-emerald-500/20 border border-emerald-500/30 px-2.5 py-1 text-[11px] font-bold text-emerald-300 hover:bg-emerald-500/30 transition disabled:opacity-40; }
    .desk-tab {
      @apply relative inline-flex items-center gap-1.5 px-4 py-3 text-[13px] font-semibold text-zinc-500 border-b-2 border-transparent hover:text-zinc-200 transition whitespace-nowrap;
    }
    .desk-tab.active { @apply text-white border-emerald-400; }
    .miner-item {
      @apply w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-left border border-transparent text-zinc-400 hover:bg-zinc-900/70 hover:text-zinc-200 transition;
    }
    .miner-item.active { @apply bg-zinc-900/90 text-white border-zinc-800 shadow-panel; }
    .miner-item-dot { @apply h-2 w-2 rounded-full bg-zinc-600 shrink-0; }
    .miner-item-dot.on { @apply bg-emerald-400; }
    .miner-item-dot.paused { @apply bg-amber-400; }
    .miner-item-dot.off { @apply bg-rose-500; }
    .miner-item-label { @apply flex flex-col gap-0.5 min-w-0 leading-tight; }
    .miner-item-name { @apply text-sm font-semibold truncate; }
    .miner-item-meta { @apply text-[10px] font-mono text-zinc-500; }
  }
  .candle-chart { cursor: grab; touch-action: none; }
  .candle-chart.dragging { cursor: grabbing; }
  [data-miner-view][hidden] { display: none; }
  [data-subtab-panel][hidden] { display: none; }
  /* Tailwind's `flex` utility beats the UA [hidden]{display:none} rule —
     without this the overlay stays painted after we set .hidden = true. */
  #rightPanelLoading[hidden] { display: none !important; }
</style>
</head>
<body class="min-h-full bg-zinc-950 text-zinc-100 font-sans antialiased">
<div class="fixed inset-0 -z-10 bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(16,185,129,0.12),transparent)]"></div>
<div class="fixed inset-0 -z-10 bg-[linear-gradient(to_right,rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:48px_48px] [mask-image:radial-gradient(ellipse_at_center,black,transparent_75%)]"></div>

<header class="sticky top-0 z-30 border-b border-zinc-800/80 bg-zinc-950/75 backdrop-blur-xl">
  <div class="w-full px-[clamp(1rem,2.5vw,2.5rem)] py-5 flex flex-wrap items-end justify-between gap-4">
    <div>
      <div class="flex items-center gap-3 mb-1">
        <span class="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 text-sm font-bold text-zinc-950 shadow-glow">89</span>
        <h1 class="text-2xl font-bold tracking-tight text-white">SN89 Desk</h1>
      </div>
      <p class="text-sm text-zinc-500 font-mono" id="walletLine">Loading wallet…</p>
    </div>
    <div class="flex items-center gap-2 text-xs text-zinc-500">
      <span class="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
      Live miner console
    </div>
  </div>
</header>

<main class="w-full px-[clamp(1rem,2.5vw,2.5rem)] py-6 pb-16 flex flex-col lg:flex-row gap-6 items-start">

<aside class="w-full lg:w-64 lg:shrink-0 lg:sticky lg:top-24 space-y-1" id="minerSidebar">
  <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500 px-2 mb-1">Miners</div>
  <button type="button" class="miner-item" id="lfSidebarItem" data-miner-kind="lf" data-miner-tag="">
    <span class="miner-item-dot" id="lfSidebarDot"></span>
    <span class="miner-item-label">
      <span class="miner-item-name">LF Manual</span>
      <span class="miner-item-meta" id="lfSidebarMeta">—</span>
    </span>
  </button>
  <div id="hfSidebarMiners" class="space-y-1">
    <span class="text-zinc-600 text-xs px-3">loading…</span>
  </div>
</aside>

<div class="flex-1 min-w-0 space-y-6">

  <div class="flex gap-1 overflow-x-auto border-b border-zinc-800/60" id="subTabNav">
    <button type="button" class="desk-tab" data-subtab="summary">Summary</button>
    <button type="button" class="desk-tab" data-subtab="open">Open Positions</button>
    <button type="button" class="desk-tab" data-subtab="signal">Live Signal</button>
    <button type="button" class="desk-tab" data-subtab="history">History</button>
  </div>

<div class="relative" id="rightPanelBody">

  <div class="absolute inset-0 z-20 flex items-center justify-center gap-2.5 rounded-2xl bg-zinc-950/70 backdrop-blur-sm text-sm text-zinc-300" id="rightPanelLoading" hidden>
    <span class="inline-block h-4 w-4 rounded-full border-2 border-zinc-600 border-t-emerald-400 animate-spin"></span>
    Loading miner…
  </div>

<div data-miner-view="lf" class="space-y-6">

  <div data-subtab-panel="summary" class="space-y-6">

    <div class="grid grid-cols-1 md:grid-cols-3 gap-4" id="timers">
      <div class="desk-timer-card">
        <div class="text-[10px] font-bold uppercase tracking-[0.12em] text-zinc-500">Next submit</div>
        <div class="text-2xl font-mono font-semibold tabular-nums mt-2 text-zinc-100" id="tGap">—</div>
        <div class="text-xs text-zinc-500 mt-1" id="tGapHint">1h gap / daily cap</div>
      </div>
      <div class="desk-timer-card">
        <div class="text-[10px] font-bold uppercase tracking-[0.12em] text-zinc-500">UTC day reset</div>
        <div class="text-2xl font-mono font-semibold tabular-nums mt-2 text-zinc-100" id="tDay">—</div>
        <div class="text-xs text-zinc-500 mt-1" id="tDayHint">quota refreshes</div>
      </div>
      <div class="desk-timer-card">
        <div class="text-[10px] font-bold uppercase tracking-[0.12em] text-zinc-500">Open call · grades in</div>
        <div class="text-2xl font-mono font-semibold tabular-nums mt-2 text-zinc-100" id="tPend">—</div>
        <div class="text-xs text-zinc-500 mt-1" id="tPendHint">pending horizon</div>
        <div class="mt-3 pt-3 border-t border-zinc-700/60 grid grid-cols-2 gap-3" id="tPriceBox" style="display:none">
          <div>
            <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Entry (T0)</div>
            <div class="font-mono text-lg font-semibold text-zinc-50 mt-0.5 tabular-nums" id="tEntry">—</div>
          </div>
          <div>
            <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Mark now</div>
            <div class="font-mono text-lg font-semibold text-emerald-300 mt-0.5 tabular-nums" id="tMark">—</div>
          </div>
        </div>
        <div class="text-xs font-mono mt-2 text-zinc-300" id="tMoveLine" style="display:none"></div>
      </div>
    </div>

    <div class="flex flex-wrap gap-2" id="scoreline">
      <span class="text-zinc-500 text-sm">Results loading…</span>
    </div>

    <section class="desk-panel p-5" id="weekPanel" style="display:none">
      <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h2 class="desk-panel-head mb-0">Past 7 days · emission window</h2>
        <span class="desk-chip" id="weekBadge">—</span>
      </div>
      <div id="weekBody" class="text-sm text-zinc-500">—</div>
    </section>

    <section class="desk-panel p-5" id="qualifyPanel">
      <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h2 class="desk-panel-head mb-0">Path to qualify</h2>
        <span class="desk-chip" id="qualifyBadge">—</span>
      </div>
      <div id="qualifyBody" class="text-sm text-zinc-500">Loading qualification path…</div>
    </section>

  </div>

  <div data-subtab-panel="open" class="space-y-6">

    <section class="desk-panel p-5" id="openCallPanel" style="display:none">
      <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h2 class="desk-panel-head mb-0">Open call · live mark</h2>
        <span class="desk-chip" id="openCallBadge">pending</span>
      </div>
      <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 text-sm" id="openCallGrid">
        <div><div class="text-[10px] uppercase tracking-wider text-zinc-500">Entry (T0)</div><div class="font-mono text-white mt-1" id="ocEntry">—</div></div>
        <div><div class="text-[10px] uppercase tracking-wider text-zinc-500">Mark now</div><div class="font-mono text-emerald-300 mt-1" id="ocMark">—</div></div>
        <div><div class="text-[10px] uppercase tracking-wider text-zinc-500">Move</div><div class="font-mono mt-1" id="ocMove">—</div></div>
        <div><div class="text-[10px] uppercase tracking-wider text-zinc-500">TP target</div><div class="font-mono text-emerald-400/90 mt-1" id="ocTp">—</div></div>
        <div><div class="text-[10px] uppercase tracking-wider text-zinc-500">SL level</div><div class="font-mono text-rose-400/90 mt-1" id="ocSl">—</div></div>
        <div><div class="text-[10px] uppercase tracking-wider text-zinc-500">To TP / SL</div><div class="font-mono text-zinc-300 mt-1" id="ocDist">—</div></div>
      </div>
      <p class="text-[11px] text-zinc-500 mt-3" id="ocNote">Binance reference; validator uses anchored tick feed (±105 bps touch ×2 ticks).</p>
    </section>
    <p class="text-sm text-zinc-500" id="lfNoOpenNote">No open LF position right now.</p>

  </div>

  <div data-subtab-panel="signal" class="space-y-6">

    <section class="desk-panel p-5 border-emerald-500/20 bg-gradient-to-br from-zinc-900/80 to-emerald-950/20" id="suggestBox" hidden>
      <div id="submitSuccessBanner" class="hidden mb-4 rounded-xl border border-emerald-500/40 bg-emerald-500/10 px-4 py-3">
        <div class="flex items-center gap-2 text-emerald-300 font-semibold">
          <span class="text-lg">✓</span>
          <span id="submitSuccessTitle">Submit successful</span>
        </div>
        <div class="text-xs font-mono text-emerald-200/80 mt-1" id="submitSuccessMeta"></div>
      </div>
      <h2 class="desk-panel-head" id="suggestHead">Selected to submit</h2>
      <div class="text-xl font-bold text-white tracking-tight" id="sugTitle">—</div>
      <div class="text-xs font-mono text-zinc-500 mt-1" id="sugMeta"></div>
      <p class="text-sm text-zinc-300 mt-3 leading-relaxed" id="sugThesis"></p>
      <div class="flex flex-wrap gap-2 items-center mt-4">
        <button class="btn-good" id="btnSubmit" disabled>Submit selected</button>
        <button class="btn-secondary" id="btnClear">Clear</button>
        <span class="font-mono text-xs" id="subStatus"></span>
      </div>
      <p class="text-xs text-zinc-500 mt-4">Pick any coin below (LONG or SHORT). Suggested top pick is only a default.</p>
    </section>

    <section class="desk-panel p-5">
      <div class="flex flex-wrap gap-3 items-center">
        <button class="btn-primary" id="btnEstimate">✦ Run Estimate</button>
        <button class="btn-secondary" id="btnRefresh">Refresh</button>
        <label class="inline-flex items-center gap-2 text-xs font-medium text-zinc-500">Mode
          <select id="mode" class="desk-select py-1.5 text-xs">
            <option value="deep" selected>Deep</option>
            <option value="fast">Fast</option>
          </select>
        </label>
        <label class="inline-flex items-center gap-2 text-xs font-medium text-zinc-500">Model
          <select id="model" class="desk-select py-1.5 text-xs min-w-[140px]">
            <option value="">default</option>
            <option value="openai/gpt-5.6-terra">gpt-5.6-terra</option>
            <option value="openai/gpt-5.6-sol">gpt-5.6-sol</option>
            <option value="openai/gpt-4o">gpt-4o</option>
            <option value="openai/gpt-4o-mini">gpt-4o-mini</option>
          </select>
        </label>
        <span class="text-xs font-mono text-zinc-500 ml-auto" id="busy"></span>
      </div>
    </section>

    <div class="grid grid-cols-1 xl:grid-cols-[2fr_1fr] gap-5">
      <section class="desk-panel p-5">
        <h2 class="desk-panel-head">Markets</h2>
        <div id="coins" class="text-zinc-500 text-sm">Loading charts…</div>
      </section>
      <section class="desk-panel p-5">
        <h2 class="desk-panel-head">Headlines</h2>
        <ul class="space-y-2 text-sm text-zinc-400 list-none p-0 m-0" id="news"><li>—</li></ul>
      </section>
    </div>

    <section class="desk-panel p-5">
      <h2 class="desk-panel-head">LLM scores</h2>
      <div id="scores" class="text-zinc-500 text-sm">Click Estimate for AI decision.</div>
    </section>

    <section class="desk-panel p-5">
      <h2 class="desk-panel-head">Raw response</h2>
      <pre id="raw" class="text-xs font-mono text-zinc-400 bg-zinc-950/80 border border-zinc-800 rounded-xl p-4 max-h-72 overflow-auto whitespace-pre-wrap">{}</pre>
    </section>

  </div>

  <div data-subtab-panel="history" class="space-y-6">

    <section class="desk-panel p-5">
      <h2 class="desk-panel-head">Submission history</h2>
      <div class="flex flex-wrap gap-2 items-center mb-4">
        <input type="search" id="histSearch" placeholder="Search pair, direction, status…" autocomplete="off" class="desk-input flex-1 min-w-[180px] max-w-md" />
        <input type="date" id="histDate" title="Filter by UTC date" class="desk-input w-auto" />
        <select id="histPageSize" title="Rows per page" class="desk-select w-auto">
          <option value="10" selected>10 / page</option>
          <option value="25">25 / page</option>
          <option value="50">50 / page</option>
        </select>
        <button type="button" class="btn-sm-ghost" id="histPrev">Prev</button>
        <button type="button" class="btn-sm-ghost" id="histNext">Next</button>
        <span class="text-xs font-mono text-zinc-500 ml-auto" id="histMeta"></span>
      </div>
      <div id="history" class="overflow-x-auto rounded-xl border border-zinc-800/60"><span class="text-zinc-500 text-sm p-4 block">Fetching…</span></div>
    </section>

  </div>

</div>

<div data-miner-view="hf" class="space-y-6" hidden>

  <section class="desk-panel p-5" id="hfAutoPanel">
    <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
      <h2 class="desk-panel-head mb-0">HF Auto (mechanism 1)</h2>
      <div class="flex items-center gap-2">
        <span class="desk-chip" id="hfRunBadge">checking…</span>
        <span class="desk-chip" id="hfQuotaBadge">—</span>
        <span class="desk-chip" id="hfPaceBadge" style="display:none">—</span>
        <span class="desk-chip" id="hfResetBadge">—</span>
        <span class="desk-chip" id="hfRotationBadge" style="display:none">—</span>
        <button class="btn-sm-ghost" id="hfToggleBtn">…</button>
      </div>
    </div>
    <p class="text-[11px] text-zinc-500">
      Tick-native automated submissions via <code>bots/hf_auto_bot.py</code> (run under PM2 as
      <code>sn89-hf-auto</code>). This panel only reads that process's state and flips its
      enable flag — start/stop the process itself from PM2.
    </p>
  </section>

  <div data-subtab-panel="summary" class="space-y-6">

    <div class="grid grid-cols-1 md:grid-cols-3 gap-4" id="hfTimers">
      <div class="desk-timer-card">
        <div class="text-[10px] font-bold uppercase tracking-[0.12em] text-zinc-500">Submit ready</div>
        <div class="text-2xl font-mono font-semibold tabular-nums mt-2 text-zinc-100" id="hfTNext">—</div>
        <div class="text-xs text-zinc-500 mt-1" id="hfTNextHint">—</div>
      </div>
      <div class="desk-timer-card">
        <div class="text-[10px] font-bold uppercase tracking-[0.12em] text-zinc-500">UTC day reset</div>
        <div class="text-2xl font-mono font-semibold tabular-nums mt-2 text-zinc-100" id="hfTDay">—</div>
        <div class="text-xs text-zinc-500 mt-1" id="hfTDayHint">quota refreshes</div>
      </div>
      <div class="desk-timer-card">
        <div class="text-[10px] font-bold uppercase tracking-[0.12em] text-zinc-500">Open call · grades in</div>
        <div class="text-2xl font-mono font-semibold tabular-nums mt-2 text-zinc-100" id="hfTPend">—</div>
        <div class="text-xs text-zinc-500 mt-1" id="hfTPendHint">pending horizon</div>
        <div class="mt-3 pt-3 border-t border-zinc-700/60 grid grid-cols-2 gap-3" id="hfTPriceBox" style="display:none">
          <div>
            <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Entry</div>
            <div class="font-mono text-lg font-semibold text-zinc-50 mt-0.5 tabular-nums" id="hfTEntry">—</div>
          </div>
          <div>
            <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Mark now</div>
            <div class="font-mono text-lg font-semibold text-emerald-300 mt-0.5 tabular-nums" id="hfTMark">—</div>
          </div>
        </div>
        <div class="text-xs font-mono mt-2 text-zinc-300" id="hfTMoveLine" style="display:none"></div>
      </div>
    </div>

    <section class="desk-panel p-5" id="hfWeekPanel" style="display:none">
      <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h2 class="desk-panel-head mb-0">Past 7 days · emission window</h2>
        <span class="desk-chip" id="hfWeekBadge">—</span>
      </div>
      <div id="hfWeekBody" class="text-sm text-zinc-500">—</div>
    </section>

    <section class="desk-panel p-5" id="hfResultsPanel">
      <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h2 class="desk-panel-head mb-0">Qualification gate &amp; results</h2>
        <div class="flex items-center gap-2 flex-wrap">
          <span class="desk-chip" id="hfResultsStatusBadge">—</span>
          <span class="desk-chip" id="hfResultsEligBadge">—</span>
          <span class="desk-chip" id="hfResultsQualBadge">—</span>
        </div>
      </div>
      <p class="text-[11px] text-zinc-500 mb-4">
        Current HF gate for hotkey <code id="hfResultsHotkey">—</code> —
        eligibility (≥50 accepted / ≥8 trading days) and <b>Wilson LB ≥ 50%</b>
        (validator qualify metric). IQ's "shrunk hit" is a separate tier metric.
      </p>
      <div id="hfResultsBody" class="text-sm text-zinc-500">Loading HF results…</div>
    </section>

  </div>

  <div data-subtab-panel="open" class="space-y-6">

    <div>
      <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">Open HF positions</div>
      <div id="hfOpen" class="overflow-x-auto rounded-xl border border-zinc-800/60">
        <span class="text-zinc-500 text-sm p-4 block">No open HF positions</span>
      </div>
    </div>

  </div>

  <div data-subtab-panel="signal" class="space-y-6">

    <div class="rounded-xl border border-zinc-800/60 bg-zinc-900/40 p-3">
      <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">Manual submit</div>
      <div class="flex flex-wrap items-end gap-2">
        <label class="flex flex-col gap-1">
          <span class="text-[10px] text-zinc-500">Pair</span>
          <select id="hfManualPair" class="desk-select py-1.5 text-xs w-32"></select>
        </label>
        <label class="flex flex-col gap-1">
          <span class="text-[10px] text-zinc-500">Direction</span>
          <select id="hfManualDir" class="desk-select py-1.5 text-xs w-28">
            <option value="LONG">LONG</option>
            <option value="SHORT">SHORT</option>
          </select>
        </label>
        <button class="btn-good" id="hfManualBtn">Submit now</button>
        <span class="font-mono text-xs" id="hfManualStatus"></span>
      </div>
      <div class="mt-3 rounded-xl border border-zinc-800/80 bg-zinc-950/50 p-3" id="hfManualReadyBox">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <span class="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Submit ready</span>
          <span class="desk-chip" id="hfManualReadyBadge">checking…</span>
        </div>
        <div class="text-2xl font-mono font-semibold tabular-nums mt-2 text-zinc-100" id="hfManualReadyTime">—</div>
        <div class="text-xs text-zinc-500 mt-1" id="hfManualReadyHint">—</div>
      </div>
      <p class="text-[10px] text-zinc-600 mt-2">
        Submits to <b>every</b> HF miner found on this box, not just the selected one. Bypasses
        scoring/streak/rotation and the daily pacing throttle, but still goes through the same
        rate-limit / pair-open / cross-mechanism gates as an automatic submission on each miner
        independently — a click that would get refused by the ingest is refused here too.
      </p>
    </div>

    <div>
      <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">Pairs · live signal</div>
      <div id="hfPairs" class="overflow-x-auto rounded-xl border border-zinc-800/60">
        <span class="text-zinc-500 text-sm p-4 block">Waiting for bot state…</span>
      </div>
    </div>

  </div>

  <div data-subtab-panel="history" class="space-y-6">

    <section class="desk-panel p-5" id="hfCallsPanel">
      <div class="flex flex-wrap items-center justify-between gap-3 mb-3">
        <h2 class="desk-panel-head mb-0">Submission ledger</h2>
        <div class="flex items-center gap-1.5" id="hfCallsFilters">
          <button class="btn-sm-ghost" data-hf-filter="all">All</button>
          <button class="btn-sm-ghost" data-hf-filter="pending">Pending</button>
          <button class="btn-sm-ghost" data-hf-filter="won">Won</button>
          <button class="btn-sm-ghost" data-hf-filter="lost">Lost</button>
          <button class="btn-sm-ghost" data-hf-filter="refused">Refused</button>
        </div>
      </div>
      <p class="text-[11px] text-zinc-500 mb-3">
        Every <code>submit_hf</code> attempt, one row per call — including refused attempts, which
        never opened a position and don't count toward the gate above.
      </p>
      <div id="hfCallsBody" class="max-h-[420px] overflow-y-auto rounded-xl border border-zinc-800/60">
        <span class="text-zinc-500 text-sm p-4 block">Loading submissions…</span>
      </div>
    </section>

    <div>
      <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">Recent decisions</div>
      <div id="hfHistory" class="overflow-x-auto rounded-xl border border-zinc-800/60">
        <span class="text-zinc-500 text-sm p-4 block">No decisions logged yet</span>
      </div>
    </div>

  </div>

</div>

</div>

</div>

</main>
<script>
const $ = (id) => document.getElementById(id);

let lastEstimate = null;
let lastMarket = null;
let statusRefreshMs = 30000;
let statusTimer = null;
const MINER_KEY = 'sn89DeskMinerV2';
const SUBTAB_KEY = 'sn89DeskSubTab';
const SUBTAB_IDS = ['summary', 'open', 'signal', 'history'];
let selectedMinerKind = 'lf';
let selectedMinerTagUI = '';
let rightPanelLoadToken = 0;

function fmtPrice(pair, p) {
  if (p == null || p === '') return '—';
  const n = Number(p);
  if (!Number.isFinite(n)) return '—';
  if (pair === 'XRPUSD') return n.toFixed(4);
  if (pair === 'TAOUSD') return n.toFixed(2);
  return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtBps(v) {
  if (v == null || v === '') return '—';
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  const sign = n >= 0 ? '+' : '';
  return `${sign}${n.toFixed(1)} bps`;
}

function bpsClass(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 'text-zinc-400';
  if (n >= 0) return 'text-emerald-400';
  return 'text-rose-400';
}

function renderOpenCall(oc) {
  const panel = $('openCallPanel');
  const priceBox = $('tPriceBox');
  const moveLine = $('tMoveLine');
  const noOpenNote = $('lfNoOpenNote');
  if (!oc || !['pending', 'sealed'].includes(String(oc.status || '').toLowerCase())) {
    if (panel) panel.style.display = 'none';
    if (priceBox) priceBox.style.display = 'none';
    if (moveLine) moveLine.style.display = 'none';
    if (noOpenNote) noOpenNote.style.display = 'block';
    return;
  }
  if (noOpenNote) noOpenNote.style.display = 'none';
  const pair = oc.trade_pair || '?';
  const dir = String(oc.direction || '').toUpperCase();
  const entryTxt = fmtPrice(pair, oc.entry_price);
  const markTxt = fmtPrice(pair, oc.mark_price);

  if (priceBox) priceBox.style.display = 'grid';
  if ($('tEntry')) $('tEntry').textContent = entryTxt;
  if ($('tMark')) $('tMark').textContent = markTxt;

  if (moveLine) {
    moveLine.style.display = 'block';
    const moveHtml = oc.move_bps != null
      ? `<span class="${bpsClass(oc.move_bps)}">${fmtBps(oc.move_bps)}</span> vs entry`
      : '';
    const distHtml = (oc.dist_tp_bps != null && oc.dist_sl_bps != null)
      ? ` · TP ${fmtBps(oc.dist_tp_bps)} left · SL ${fmtBps(oc.dist_sl_bps)} buffer`
      : '';
    moveLine.innerHTML = `${pair} ${dir} · ${moveHtml}${distHtml}`;
  }

  if (panel) panel.style.display = 'block';
  if ($('openCallBadge')) $('openCallBadge').textContent = `${pair} ${dir} · ${oc.status}`;
  if ($('ocEntry')) $('ocEntry').textContent = entryTxt;
  if ($('ocMark')) $('ocMark').textContent = markTxt;
  if ($('ocMove')) {
    $('ocMove').textContent = fmtBps(oc.move_bps);
    $('ocMove').className = `font-mono mt-1 ${bpsClass(oc.move_bps)}`;
  }
  if ($('ocTp')) $('ocTp').textContent = oc.tp_price != null ? `${fmtPrice(pair, oc.tp_price)} (+${oc.tp_bps || '?'} bps)` : '—';
  if ($('ocSl')) $('ocSl').textContent = oc.sl_price != null ? `${fmtPrice(pair, oc.sl_price)} (-${oc.sl_bps || '?'} bps)` : '—';
  if ($('ocDist')) {
    if (oc.dist_tp_bps != null && oc.dist_sl_bps != null) {
      $('ocDist').innerHTML = `<span class="text-emerald-400">${fmtBps(oc.dist_tp_bps)}</span> / <span class="text-rose-400">${fmtBps(oc.dist_sl_bps)}</span>`;
    } else {
      $('ocDist').textContent = '—';
    }
  }
  if ($('ocNote')) $('ocNote').textContent = oc.price_note || 'Binance reference; validator uses anchored tick feed.';
}

let pending = null;
let statusSnap = null;
let allCalls = [];
let historyState = { page: 1, pageSize: 10, query: '', date: '' };
let chartState = {};
let submitSuccess = null;
let ends = { gap: null, day: null, pend: null, pendLabel: '', nextMode: 'open' };

const BADGE = {
  long: 'inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-300 ring-1 ring-inset ring-emerald-500/25',
  short: 'inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-rose-500/10 text-rose-300 ring-1 ring-inset ring-rose-500/25',
  none: 'inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-zinc-800 text-zinc-400 ring-1 ring-inset ring-zinc-700',
  won: 'inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-300 ring-1 ring-inset ring-emerald-500/25',
  lost: 'inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-rose-500/10 text-rose-300 ring-1 ring-inset ring-rose-500/25',
  pending: 'inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-amber-500/10 text-amber-200 ring-1 ring-inset ring-amber-500/25',
  sealed: 'inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-amber-500/10 text-amber-200 ring-1 ring-inset ring-amber-500/25',
  washed: 'inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-zinc-500/10 text-zinc-300 ring-1 ring-inset ring-zinc-500/25',
  void: 'inline-flex items-center rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-zinc-800 text-zinc-500 ring-1 ring-inset ring-zinc-700',
};
const TW_VAL = 'text-2xl font-mono font-semibold tabular-nums mt-2';

function setTimerVal(el, text, state) {
  el.textContent = text;
  el.className = TW_VAL + (
    state === 'ok' ? ' text-emerald-400'
    : state === 'wait' ? ' text-amber-400'
    : state === 'muted' ? ' text-zinc-500'
    : ' text-zinc-100'
  );
}

function setSubStatus(text, kind) {
  $('subStatus').textContent = text;
  $('subStatus').className = 'font-mono text-xs' + (
    kind === 'err' ? ' text-rose-400' : kind === 'ok' ? ' text-emerald-400' : kind === 'wait' ? ' text-amber-400' : ' text-zinc-500'
  );
}

function clearSubmitSuccess() {
  submitSuccess = null;
  $('submitSuccessBanner').classList.add('hidden');
  $('suggestBox').classList.remove('border-emerald-400/50', 'shadow-glow');
  $('suggestHead').textContent = 'Selected to submit';
  $('btnSubmit').textContent = 'Submit selected';
}

function renderSubmitSuccessPanel(apiData) {
  if (!submitSuccess) return;
  const s = submitSuccess;
  $('suggestBox').hidden = false;
  $('suggestBox').classList.add('border-emerald-400/50', 'shadow-glow');
  $('submitSuccessBanner').classList.remove('hidden');
  $('submitSuccessTitle').textContent = `Submit successful — ${s.pair} ${s.direction}`;
  const via = (apiData && apiData.via) || s.via || 'miner';
  const when = new Date(s.at).toISOString().slice(11, 19) + ' UTC';
  $('submitSuccessMeta').textContent = `Committed via ${via} · ${when}`;
  $('suggestHead').textContent = 'Last submit';
  $('sugTitle').textContent = `${s.pair} ${s.direction}`;
  $('sugMeta').textContent = s.thesis ? 'thesis saved on-chain' : 'manual submit';
  $('sugThesis').textContent = s.thesis || '';
  $('btnSubmit').textContent = 'Submitted ✓';
  $('btnSubmit').disabled = true;
  const gap = statusSnap && statusSnap.timers && statusSnap.timers.gap_left_s;
  const q = statusSnap && statusSnap.quota;
  if (gap > 0) {
    setSubStatus(`Success · next submit in ${fmtDur(gap)} (${(q && q.today_used) || '?'}/${(q && q.max_per_day) || 3} today)`, 'ok');
  } else {
    setSubStatus('Success · on-chain commit confirmed', 'ok');
  }
}

function deskTable(headHtml, bodyHtml) {
  return `<div class="overflow-x-auto rounded-xl border border-zinc-800/60"><table class="desk-table"><thead>${headHtml}</thead><tbody>${bodyHtml}</tbody></table></div>`;
}

function fmtDur(s) {
  s = Math.max(0, Math.floor(Number(s) || 0));
  if (s <= 0) return 'ready';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2,'0')}m ${String(sec).padStart(2,'0')}s`;
  return `${m}m ${String(sec).padStart(2,'0')}s`;
}

function pill(dir) {
  const d = (dir || 'NONE').toUpperCase();
  const k = d === 'LONG' ? 'long' : d === 'SHORT' ? 'short' : 'none';
  return `<span class="${BADGE[k]}">${d}</span>`;
}

function statusTag(st) {
  const s = (st || '').toLowerCase();
  return `<span class="${BADGE[s] || BADGE.none}">${st || '—'}</span>`;
}

function sparklineSvg(candles, w = 200, h = 52) {
  if (!candles || !candles.length) return '<span class="text-zinc-500 text-xs">No candle data</span>';
  const closes = candles.map(c => Number(c.c)).filter(v => Number.isFinite(v));
  if (closes.length < 2) return '<span class="text-zinc-500 text-xs">Not enough bars</span>';
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const range = max - min || 1;
  const pts = closes.map((c, i) => {
    const x = (i / (closes.length - 1)) * w;
    const y = h - ((c - min) / range) * (h - 6) - 3;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const up = closes[closes.length - 1] >= closes[0];
  const stroke = up ? '#34d399' : '#f87171';
  const fill = up ? 'rgba(52,211,153,0.12)' : 'rgba(248,113,113,0.12)';
  const area = `0,${h} ${pts} ${w},${h}`;
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
    <polygon points="${area}" fill="${fill}"></polygon>
    <polyline fill="none" stroke="${stroke}" stroke-width="1.6" points="${pts}"></polyline>
  </svg>`;
}

function fmtPrice(v) {
  v = Number(v);
  if (!Number.isFinite(v)) return '—';
  if (Math.abs(v) >= 1000) return v.toFixed(1);
  if (Math.abs(v) >= 10) return v.toFixed(2);
  return v.toFixed(4);
}

function candleChartBlock(pair, candles) {
  const prev = chartState[pair];
  const total = (candles || []).length;
  const start = prev && prev.candles && prev.candles.length === total
    ? Math.min(prev.start, total - 8)
    : Math.max(0, total - 32);
  chartState[pair] = {
    candles: candles || [],
    start: Math.max(0, start),
    end: total,
    drag: null,
  };
  return `<div class="chart-wrap" data-pair="${pair}">
    <div class="flex flex-wrap gap-1.5 items-center mb-2">
      <button type="button" class="btn-sm-ghost chart-range" data-pair="${pair}" data-n="24">6h</button>
      <button type="button" class="btn-sm-ghost chart-range" data-pair="${pair}" data-n="48">12h</button>
      <button type="button" class="btn-sm-ghost chart-range" data-pair="${pair}" data-n="all">All</button>
      <span class="text-[10px] font-mono text-zinc-500 ml-auto">drag pan · scroll zoom</span>
    </div>
    <svg class="candle-chart block w-full rounded-xl border border-zinc-800 bg-zinc-950/80 h-[clamp(140px,16vw,220px)]" data-pair="${pair}" viewBox="0 0 400 140" preserveAspectRatio="none"></svg>
    <div class="chart-tip text-[10px] font-mono text-zinc-500 mt-1 min-h-[1rem]" data-pair="${pair}">—</div>
  </div>`;
}

function chartTip(pair) {
  const wrap = document.querySelector(`.chart-wrap[data-pair="${CSS.escape(pair)}"]`);
  return wrap ? wrap.querySelector('.chart-tip') : null;
}

function drawCandleChart(pair) {
  const st = chartState[pair];
  const svg = document.querySelector(`svg.candle-chart[data-pair="${CSS.escape(pair)}"]`);
  if (!st || !svg || !st.candles.length) return;
  const slice = st.candles.slice(st.start, st.end);
  const W = 400, H = 140, padT = 8, padB = 16, padL = 46, padR = 8;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  if (!slice.length) {
    svg.innerHTML = '';
    return;
  }
  const highs = slice.map(c => Number(c.h));
  const lows = slice.map(c => Number(c.l));
  const yMin = Math.min(...lows);
  const yMax = Math.max(...highs);
  const yRange = yMax - yMin || 1;
  const yScale = v => padT + plotH - ((v - yMin) / yRange) * plotH;
  const n = slice.length;
  const slot = plotW / n;
  const bodyW = Math.max(1.4, slot * 0.58);
  const parts = [];
  for (let i = 0; i <= 3; i++) {
    const y = padT + (plotH * i / 3);
    const price = yMax - (yRange * i / 3);
    parts.push(`<line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" stroke="#26262e" stroke-width="0.6"/>`);
    parts.push(`<text x="${padL - 4}" y="${(y + 3).toFixed(1)}" fill="#93939f" font-size="7" text-anchor="end">${fmtPrice(price)}</text>`);
  }
  slice.forEach((c, i) => {
    const x = padL + i * slot + slot / 2;
    const o = Number(c.o), h = Number(c.h), l = Number(c.l), cl = Number(c.c);
    const up = cl >= o;
    const col = up ? '#34d399' : '#f87171';
    const yO = yScale(o), yC = yScale(cl), yH = yScale(h), yL = yScale(l);
    const top = Math.min(yO, yC);
    const bh = Math.max(1, Math.abs(yC - yO));
    parts.push(`<line x1="${x.toFixed(1)}" y1="${yH.toFixed(1)}" x2="${x.toFixed(1)}" y2="${yL.toFixed(1)}" stroke="${col}" stroke-width="1"/>`);
    parts.push(`<rect x="${(x - bodyW / 2).toFixed(1)}" y="${top.toFixed(1)}" width="${bodyW.toFixed(1)}" height="${bh.toFixed(1)}" fill="${col}" opacity="0.88"/>`);
  });
  svg.innerHTML = parts.join('');
  const tip = chartTip(pair);
  if (tip && !(chartState[pair] && chartState[pair].drag)) {
    const bars = st.end - st.start;
    tip.textContent = `${pair} · bars ${st.start + 1}–${st.end} of ${st.candles.length} (${bars} visible)`;
  }
}

function setChartRange(pair, n) {
  const st = chartState[pair];
  if (!st) return;
  const total = st.candles.length;
  if (n === 'all') {
    st.start = 0;
    st.end = total;
  } else {
    const count = Math.min(total, Number(n) || 32);
    st.end = total;
    st.start = Math.max(0, total - count);
  }
  drawCandleChart(pair);
}

function bindCandleChart(pair) {
  const svg = document.querySelector(`svg.candle-chart[data-pair="${CSS.escape(pair)}"]`);
  if (!svg || svg.dataset.bound === '1') return;
  svg.dataset.bound = '1';
  const tip = chartTip(pair);

  svg.addEventListener('wheel', (e) => {
    e.preventDefault();
    const st = chartState[pair];
    if (!st) return;
    const len = st.end - st.start;
    const rect = svg.getBoundingClientRect();
    const ratio = rect.width > 0 ? (e.clientX - rect.left) / rect.width : 0.5;
    const zoomIn = e.deltaY < 0;
    const newLen = Math.max(
      8,
      Math.min(st.candles.length, Math.round(len * (zoomIn ? 0.82 : 1.18))),
    );
    const anchor = st.start + Math.round(len * ratio);
    st.start = Math.max(0, Math.min(st.candles.length - newLen, anchor - Math.round(newLen * ratio)));
    st.end = st.start + newLen;
    drawCandleChart(pair);
  }, { passive: false });

  svg.addEventListener('mousedown', (e) => {
    const st = chartState[pair];
    if (!st) return;
    st.drag = { x: e.clientX, start: st.start, end: st.end };
    svg.classList.add('dragging');
  });

  const endDrag = () => {
    const st = chartState[pair];
    if (st) st.drag = null;
    svg.classList.remove('dragging');
  };

  svg.addEventListener('mousemove', (e) => {
    const st = chartState[pair];
    if (!st) return;
    if (st.drag) {
      const len = st.drag.end - st.drag.start;
      const dx = e.clientX - st.drag.x;
      const shift = Math.round(-dx / 6);
      st.start = Math.max(0, Math.min(st.candles.length - len, st.drag.start + shift));
      st.end = st.start + len;
      drawCandleChart(pair);
      return;
    }
    if (!tip) return;
    const rect = svg.getBoundingClientRect();
    const ratio = rect.width > 0 ? (e.clientX - rect.left) / rect.width : 0;
    const idx = st.start + Math.min(st.end - st.start - 1, Math.max(0, Math.floor(ratio * (st.end - st.start))));
    const c = st.candles[idx];
    if (!c) return;
    const t = c.t ? new Date(Number(c.t)).toISOString().slice(11, 16) + ' UTC' : `#${idx + 1}`;
    tip.textContent = `${t} · O ${fmtPrice(c.o)} H ${fmtPrice(c.h)} L ${fmtPrice(c.l)} C ${fmtPrice(c.c)}`;
  });
  svg.addEventListener('mouseup', endDrag);
  svg.addEventListener('mouseleave', () => {
    endDrag();
    drawCandleChart(pair);
  });
}

function initCoinCharts() {
  document.querySelectorAll('svg.candle-chart[data-pair]').forEach(svg => {
    const pair = svg.getAttribute('data-pair');
    drawCandleChart(pair);
    bindCandleChart(pair);
  });
  document.querySelectorAll('button.chart-range').forEach(btn => {
    btn.onclick = () => setChartRange(btn.getAttribute('data-pair'), btn.getAttribute('data-n'));
  });
}

function pnlBarsBlock(pair) {
  const rows = callsForPair(pair).slice().reverse();
  const graded = rows.filter(c =>
    c.outcome_bps != null || ['won', 'lost', 'washed', 'pending', 'sealed'].includes(String(c.status || '').toLowerCase()),
  );
  if (!graded.length) return '';
  const W = 240, H = 78, padX = 8, padY = 10, baseY = H - padY - 8;
  const maxAbs = Math.max(105, ...graded.map(c => Math.abs(Number(c.outcome_bps) || 0)));
  const slot = (W - padX * 2) / graded.length;
  const barW = Math.max(6, Math.min(20, slot * 0.72));
  let cum = 0;
  const parts = [];
  parts.push(`<line x1="${padX}" y1="${baseY}" x2="${W - padX}" y2="${baseY}" stroke="#333342" stroke-width="0.8"/>`);
  graded.forEach((c, i) => {
    const bps = Number(c.outcome_bps);
    const st = String(c.status || '').toLowerCase();
    const hasBps = Number.isFinite(bps);
    cum += hasBps ? bps : 0;
    const x = padX + i * slot + slot / 2;
    let col = '#fde68a';
    let h = 6;
    if (hasBps && bps > 0) { col = '#34d399'; h = Math.max(4, (Math.abs(bps) / maxAbs) * (baseY - padY - 8)); }
    else if (hasBps && bps < 0) { col = '#f87171'; h = Math.max(4, (Math.abs(bps) / maxAbs) * (baseY - padY - 8)); }
    else if (st === 'washed') { col = '#93939f'; h = 5; }
    else if (st === 'pending' || st === 'sealed') { col = '#fbbf24'; h = 8; }
    const y = bps >= 0 || !hasBps ? baseY - h : baseY;
    parts.push(`<rect x="${(x - barW / 2).toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" fill="${col}" opacity="0.9" rx="1"><title>${(c.t0_utc || '').replace(' UTC', '')} ${c.direction || ''} ${st}${hasBps ? ' · ' + bps + ' bps' : ''}</title></rect>`);
  });
  const cumCls = cum >= 0 ? 'text-emerald-400' : 'text-rose-400';
  const wins = graded.filter(c => c.status === 'won').length;
  const losses = graded.filter(c => c.status === 'lost').length;
  return `<div class="mt-3">
    <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">Outcome PnL · ${pair}</div>
    <svg class="block w-full h-[78px] rounded-xl border border-zinc-800 bg-zinc-950/80" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${parts.join('')}</svg>
    <div class="flex flex-wrap gap-3 text-[11px] text-zinc-500 mt-2">
      <span><b class="font-mono ${cumCls}">${cum >= 0 ? '+' : ''}${cum}</b> bps cum</span>
      <span>${wins}W / ${losses}L</span>
      <span>oldest → newest</span>
    </div>
  </div>`;
}

function callsForPair(pair) {
  const p = String(pair || '').toUpperCase();
  return (allCalls || []).filter(c => String(c.trade_pair || '').toUpperCase() === p);
}

function coinHistoryBlock(pair) {
  const rows = callsForPair(pair);
  if (!rows.length) {
    return '<span class="text-zinc-500 text-xs">No SN89 calls for this pair yet</span>';
  }
  const won = rows.filter(c => c.status === 'won').length;
  const lost = rows.filter(c => c.status === 'lost').length;
  const washed = rows.filter(c => c.status === 'washed').length;
  const open = rows.filter(c => c.status === 'pending' || c.status === 'sealed').length;
  const body = rows.slice(0, 5).map(c => {
    const bps = c.outcome_bps != null ? `${c.outcome_bps} bps` : '—';
    return `<tr class="hover:bg-zinc-900/40">
      <td class="desk-td font-mono text-xs">${(c.t0_utc || '—').replace(' UTC', '')}</td>
      <td class="desk-td">${pill(c.direction)}</td>
      <td class="desk-td">${statusTag(c.status)}</td>
      <td class="desk-td font-mono text-xs">${bps}</td>
    </tr>`;
  }).join('');
  return `
    <div class="text-xs text-zinc-500 mb-2">${won}W · ${lost}L · ${washed} wash · ${open} open</div>
    ${pnlBarsBlock(pair)}
    <table class="desk-table text-xs mt-2"><thead><tr><th class="desk-th">When</th><th class="desk-th">Dir</th><th class="desk-th">Result</th><th class="desk-th">Bps</th></tr></thead>
    <tbody>${body}</tbody></table>`;
}

function expectationBlock(score, coin) {
  const s = score || {};
  const dir = String(s.direction || 'NONE').toUpperCase();
  const conf = s.confidence != null ? Number(s.confidence) : null;
  const wash = s.wash_risk != null ? Number(s.wash_risk) : null;
  const thesis = s.thesis || s.why || '';
  const reasoning = s.reasoning || s.tech_reason || '';
  const action = dir === 'LONG' || dir === 'SHORT' ? dir : 'NONE (sit out)';
  const tp = coin && coin.tp_sl_bps != null ? `${coin.tp_sl_bps} bps TP/SL` : '—';
  const reach = coin && coin.reach_score != null ? `reach ${coin.reach_score}` : '';
  const atr = coin && coin.atr_to_tp_ratio != null ? `ATR/TP ${coin.atr_to_tp_ratio}` : '';
  return `
    <div class="text-sm">${pill(dir === 'LONG' || dir === 'SHORT' ? dir : 'NONE')}
      <span class="font-mono text-zinc-300">${action}</span>
      ${conf != null ? `<span class="font-mono text-zinc-500"> · conf ${conf.toFixed(2)}</span>` : ''}
      ${wash != null ? `<span class="font-mono text-zinc-500"> · wash ${wash.toFixed(2)}</span>` : ''}
    </div>
    <div class="text-xs font-mono text-zinc-500 mt-1">${tp}${reach ? ' · ' + reach : ''}${atr ? ' · ' + atr : ''}</div>
    ${thesis ? `<div class="text-sm text-zinc-300 mt-2 leading-relaxed">${thesis}</div>` : ''}
    ${reasoning ? `<div class="text-xs text-zinc-500 mt-2 whitespace-pre-wrap max-h-24 overflow-auto leading-relaxed">${reasoning}</div>` : '<div class="text-xs text-zinc-500 mt-2">Run Estimate for LLM thesis</div>'}`;
}

function filterHistoryCalls(calls) {
  const q = historyState.query.trim().toLowerCase();
  const date = historyState.date.trim();
  return (calls || []).filter(c => {
    if (date) {
      const day = (c.t0_utc || '').slice(0, 10);
      if (day !== date) return false;
    }
    if (!q) return true;
    const hay = [
      c.trade_pair, c.direction, c.status, c.exit_reason, c.void_reason, c.t0_utc,
      c.outcome_bps != null ? String(c.outcome_bps) : '',
    ].join(' ').toLowerCase();
    return hay.includes(q);
  });
}

function renderHistoryPage() {
  const filtered = filterHistoryCalls(allCalls);
  const total = filtered.length;
  const pages = Math.max(1, Math.ceil(total / historyState.pageSize));
  if (historyState.page > pages) historyState.page = pages;
  if (historyState.page < 1) historyState.page = 1;
  const start = (historyState.page - 1) * historyState.pageSize;
  const slice = filtered.slice(start, start + historyState.pageSize);

  $('histMeta').textContent = total
    ? `${start + 1}–${start + slice.length} of ${total} · page ${historyState.page}/${pages}`
    : (historyState.query || historyState.date ? '0 matches' : '');
  $('histPrev').disabled = historyState.page <= 1;
  $('histNext').disabled = historyState.page >= pages;

  if (!allCalls.length) {
    $('history').innerHTML = statusSnap && statusSnap.iq_ok === false
      ? '<span class="text-rose-400 text-sm p-4 block">Could not load IQ history</span>'
      : '<span class="text-zinc-500 text-sm p-4 block">No calls yet</span>';
    return;
  }
  if (!filtered.length) {
    $('history').innerHTML = '<span class="text-zinc-500 text-sm p-4 block">No rows match your search</span>';
    return;
  }
  $('history').innerHTML = deskTable(`<tr>
      <th class="desk-th whitespace-nowrap">When</th><th class="desk-th">Call</th><th class="desk-th whitespace-nowrap">Entry</th><th class="desk-th whitespace-nowrap">Mark</th><th class="desk-th">Move</th><th class="desk-th">Result</th><th class="desk-th">Bps</th><th class="desk-th">Exit</th><th class="desk-th">Timer</th>
    </tr>`, slice.map(c => {
      const left = c.horizon_left_s;
      const open = c.status === 'pending' || c.status === 'sealed';
      const timer = open
        ? (left > 0 ? fmtDur(left) + ' left' : 'window ending')
        : '—';
      const pair = c.trade_pair || '?';
      const entryCell = c.entry_price != null
        ? `<span class="text-zinc-100 font-semibold">${fmtPrice(pair, c.entry_price)}</span>`
        : (open ? '<span class="text-zinc-600">…</span>' : '—');
      const markCell = c.mark_price != null
        ? `<span class="text-emerald-300">${fmtPrice(pair, c.mark_price)}</span>`
        : (open ? '<span class="text-zinc-600">…</span>' : '—');
      const moveCell = c.move_bps != null
        ? `<span class="${bpsClass(c.move_bps)}">${fmtBps(c.move_bps)}</span>`
        : (open ? '—' : '—');
      return `<tr class="hover:bg-zinc-900/40">
        <td class="desk-td font-mono text-xs">${c.t0_utc || '—'}</td>
        <td class="desk-td font-mono text-xs"><b class="text-white">${pair}</b> ${pill(c.direction)}</td>
        <td class="desk-td font-mono text-xs">${entryCell}</td>
        <td class="desk-td font-mono text-xs text-emerald-300/90">${markCell}</td>
        <td class="desk-td font-mono text-xs">${moveCell}</td>
        <td class="desk-td">${statusTag(c.status)}</td>
        <td class="desk-td font-mono text-xs">${c.outcome_bps != null ? c.outcome_bps : '—'}</td>
        <td class="desk-td text-zinc-500 text-xs">${c.exit_reason || c.void_reason || '—'}</td>
        <td class="desk-td font-mono text-xs ${left > 0 ? 'text-amber-400' : ''}">${timer}</td>
      </tr>`;
    }).join(''));
}

function paintTimers() {
  const now = Date.now() / 1000;
  tickHfSubmitReady();
  if (ends.nextMode === 'day' && ends.day != null) {
    const left = ends.day - now;
    setTimerVal($('tGap'), left > 0 ? fmtDur(left) : 'ready', left > 0 ? 'wait' : 'ok');
  } else if (ends.nextMode === 'gap' && ends.gap != null) {
    const left = ends.gap - now;
    setTimerVal($('tGap'), left > 0 ? fmtDur(left) : 'ready', left > 0 ? 'wait' : 'ok');
  } else {
    setTimerVal($('tGap'), 'ready', 'ok');
  }
  if (ends.day != null) {
    setTimerVal($('tDay'), fmtDur(ends.day - now), '');
  }
  if (ends.pend != null) {
    const left = ends.pend - now;
    setTimerVal($('tPend'), left > 0 ? fmtDur(left) : 'due', left > 0 ? 'wait' : '');
    $('tPendHint').textContent = ends.pendLabel || 'pending horizon';
  } else {
    setTimerVal($('tPend'), 'none', 'muted');
    $('tPendHint').textContent = 'no open call';
  }
}

function gateCheck(ok, label, detail) {
  const cls = ok ? 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10' : 'text-amber-300 border-amber-500/30 bg-amber-500/10';
  const mark = ok ? '✓' : '○';
  return `<div class="rounded-xl border px-3 py-2.5 ${cls}">
    <div class="text-[10px] font-bold uppercase tracking-wider opacity-80">${label}</div>
    <div class="font-mono text-sm mt-1">${mark} ${detail}</div>
  </div>`;
}

// One week IS the whole emission horizon on both mechanisms, so this block —
// not the lifetime scoreline above it — is the window today's weight is sized
// from. A win that ages past the window contributes exactly zero.
//
// "Decay-weighted" here is an UNWEIGHTED proxy: same linear curve, same window,
// same win cap as the validator's tally, but every win counts 1.0 because the
// per-win tier and wash-efficiency multipliers are not in the call feed. It
// tracks the shape of the real number, not its level — and the real number is
// then normalized across the field, so it is a numerator either way.
function renderWeek(panelId, badgeId, bodyId, w) {
  const panel = $(panelId);
  if (!panel) return;
  if (!w) { panel.style.display = 'none'; return; }
  panel.style.display = '';

  const approx = w.qualified_approx ? '~' : '';
  const stale = w.won === 0;

  // Track the constant rather than hardcoding 7 — SN89_EMISSION_DECAY_S and
  // SN89_HF_EMISSION_DECAY_S can both be overridden, and a heading that lies
  // about the window is worse than no heading.
  const head = panel.querySelector('h2');
  if (head) head.textContent = `Past ${w.window_days} days · emission window`;

  $(badgeId).textContent = stale ? 'no live wins' : 'rolling';
  $(badgeId).className = stale
    ? 'desk-chip border-amber-500/40 text-amber-200 bg-amber-500/10'
    : 'desk-chip border-emerald-500/40 text-emerald-300 bg-emerald-500/10';

  const tile = (label, value, cls, title) => `<div class="rounded-xl border border-zinc-800/60 px-3 py-2.5" ${title ? `title="${title}"` : ''}>
    <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500">${label}</div>
    <div class="font-mono text-lg mt-1 ${cls || 'text-zinc-200'}">${value}</div>
  </div>`;

  const tiles = [
    tile('Wins', w.won, w.won > 0 ? 'text-emerald-300' : 'text-zinc-500'),
    tile('Qualified', `${approx}${w.qualified_won}`,
      w.qualified_won > 0 ? 'text-emerald-300' : 'text-amber-300',
      'Wins that passed the qualify gate as of their own t0 — only these earn'),
    tile('Decay-weighted', w.decay_sum_qualified.toFixed(2),
      w.decay_sum_qualified > 0 ? 'text-emerald-300' : 'text-zinc-500',
      `Unweighted proxy: sum of (1 - age/${w.window_days}d) over qualified wins. All wins: ${w.decay_sum.toFixed(2)}`),
    tile('Losses', w.lost, w.lost > 0 ? 'text-rose-300' : 'text-zinc-500'),
    tile('Wash', w.washed, 'text-zinc-400'),
    tile('Newest win', w.newest_win_age_h != null ? `${w.newest_win_age_h}h ago` : '—',
      w.newest_win_age_h != null ? 'text-zinc-300' : 'text-zinc-500'),
  ].join('');

  const notes = [];
  if (!w.eligible) {
    notes.push('<span class="text-amber-300">Not eligible yet — no win earns until the gate opens.</span>');
  }
  if (stale) {
    notes.push(`<span class="text-amber-300">No wins in the last ${w.window_days} days — the decayed tally behind your weight is zero regardless of lifetime record.</span>`);
  }
  if (w.cap_binding) {
    notes.push(`Win cap binding: ${w.won} live wins, only the most recent ${w.win_cap} count.`);
  }
  if (w.qualified_approx) {
    notes.push('<b>~</b> approximate: the call feed is paged and does not reach back a full reputation window, so the as-of gate is reconstructed with your IQ win/loss totals folded in as an older-history seed.');
  }
  notes.push(`Rolling from right now, <b>not a calendar week</b> — nothing resets at UTC midnight here, each win simply drops out ${w.window_days * 24}h after it was posted.`);
  notes.push(`Decay-weighted is an <b>unweighted proxy</b> — same curve and window as the validator's tally, but without the per-win tier and wash-efficiency multipliers, which the call feed does not carry. It is also a numerator: your actual share is this normalized across the field.`);

  $(bodyId).innerHTML = `
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">${tiles}</div>
    <p class="text-[11px] text-zinc-500 mt-3 leading-relaxed">${notes.join('<br>')}</p>`;
}

function renderQualifyPath(qp) {
  if (!qp) {
    $('qualifyBadge').textContent = 'IQ API unavailable';
    $('qualifyBadge').className = 'desk-chip border-zinc-700 text-zinc-500';
    $('qualifyBody').innerHTML = '<span class="text-zinc-500">Could not load miner stats from InfiniteQuant API.</span>';
    return;
  }
  const cur = qp.current || {};
  const gaps = qp.gaps || {};
  const th = qp.thresholds || {};
  const qualified = !!qp.meets_gate || !!cur.qualifies;
  $('qualifyBadge').textContent = qualified ? 'Qualified' : 'Not qualified';
  $('qualifyBadge').className = qualified
    ? 'desk-chip border-emerald-500/40 text-emerald-300 bg-emerald-500/10'
    : 'desk-chip border-amber-500/40 text-amber-200 bg-amber-500/10';

  const decisivesLeft = gaps.decisives_left ?? 0;
  const lbGap = gaps.lb_gap_pct ?? 0;
  const minTrades = qp.min_trades_to_qualify;
  const minWins = qp.min_wins_needed;
  let hint = '';
  if (!qualified) {
    if (minTrades != null && minWins != null) {
      hint = `Fastest modeled path: <b class="text-zinc-200">${minTrades}</b> more decisive trade${minTrades === 1 ? '' : 's'}, including <b class="text-emerald-300">${minWins}</b> win${minWins === 1 ? '' : 's'} (no further losses).`;
    } else {
      hint = 'Keep stacking decisive wins — LB rises slowly with a thin sample.';
    }
  } else {
    hint = 'All three gates pass. Stay above thresholds to keep emissions active.';
  }

  const checks = [
    gateCheck(gaps.decisive_ok, 'Decisive', `${cur.decisive ?? '—'}/${th.min_decisive}${decisivesLeft ? ` · ${decisivesLeft} left` : ''}`),
    gateCheck(gaps.hit_ok, 'Hit rate', `${cur.hit_pct ?? '—'}% / ${th.min_hit_pct}%`),
    gateCheck(gaps.lb_ok, 'Confidence LB', `${cur.lb_pct ?? '—'}% / ${th.lb_floor_pct}%${lbGap ? ` · ${lbGap}% gap` : ''}`),
  ].join('');

  const rows = (qp.scenarios || []).map(s => {
    const ok = !!s.qualifies;
    const rowCls = ok ? 'bg-emerald-500/5' : '';
    const qual = ok ? '<span class="text-emerald-400 font-semibold">yes</span>' : '<span class="text-zinc-500">no</span>';
    return `<tr class="${rowCls}">
      <td class="desk-td font-medium text-zinc-200">${s.label}</td>
      <td class="desk-td font-mono text-xs">${s.path || '—'}</td>
      <td class="desk-td font-mono text-xs">${s.wins}W/${s.losses}L</td>
      <td class="desk-td font-mono text-xs">${s.hit_pct}%</td>
      <td class="desk-td font-mono text-xs">${s.lb_pct}%</td>
      <td class="desk-td">${qual}</td>
    </tr>`;
  }).join('');

  $('qualifyBody').innerHTML = `
    <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">${checks}</div>
    <p class="text-xs text-zinc-500 mb-4 leading-relaxed">${hint}</p>
    <div class="overflow-x-auto rounded-xl border border-zinc-800/60">
      <table class="desk-table">
        <thead><tr>
          <th class="desk-th">Scenario</th>
          <th class="desk-th">Path</th>
          <th class="desk-th">Record</th>
          <th class="desk-th">Hit</th>
          <th class="desk-th">LB</th>
          <th class="desk-th">Qualify</th>
        </tr></thead>
        <tbody>${rows || '<tr><td class="desk-td text-zinc-500" colspan="6">No scenarios</td></tr>'}</tbody>
      </table>
    </div>
    <p class="text-[10px] text-zinc-600 mt-3 font-mono">LB scenarios use Wilson lower bound calibrated to IQ API · washes never count</p>`;
}

function renderStatus(data) {
  statusSnap = data;
  const q = data.quota || {};
  const sc = data.score || {};
  const now = data.now_unix || (Date.now() / 1000);
  $('walletLine').textContent = `${data.wallet || '—'} · ${(data.hotkey || '').slice(0, 8)}…${(data.hotkey || '').slice(-4)} · serve ${data.serve && data.serve.up ? 'up' : 'down'}`;

  const lfDot = $('lfSidebarDot');
  if (lfDot) lfDot.className = `miner-item-dot ${data.submit_allowed ? 'on' : 'paused'}`;
  const lfMeta = $('lfSidebarMeta');
  if (lfMeta) lfMeta.textContent = `${q.today_used ?? 0}/${q.max_per_day ?? '—'} today · ${data.submit_allowed ? 'open' : 'blocked'}`;

  const gap = Number(data.timers && data.timers.gap_left_s || 0);
  const day = Number(data.timers && data.timers.day_reset_s || 0);
  ends.gap = gap > 0 ? now + gap : now;
  ends.day = now + day;
  const reason = String(data.submit_block_reason || '').toLowerCase();
  if (!data.submit_allowed && (reason.includes('daily cap') || (q.today_used >= q.max_per_day))) {
    ends.nextMode = 'day';
    $('tGapHint').textContent = `daily cap ${q.today_used}/${q.max_per_day} · until UTC midnight`;
  } else if (!data.submit_allowed && gap > 0) {
    ends.nextMode = 'gap';
    $('tGapHint').textContent = `min 1h gap · quota ${q.today_used}/${q.max_per_day}`;
  } else if (data.submit_allowed) {
    ends.nextMode = 'open';
    $('tGapHint').textContent = `quota ${q.today_used}/${q.max_per_day} · can submit`;
  } else {
    ends.nextMode = gap > 0 ? 'gap' : 'day';
    $('tGapHint').textContent = data.submit_block_reason || `quota ${q.today_used}/${q.max_per_day}`;
  }
  const pend = (data.timers && data.timers.pending) || [];
  if (pend.length) {
    const p = pend[0];
    ends.pend = p.horizon_end_unix;
    ends.pendLabel = `${p.trade_pair} ${p.direction} · ${p.status}`;
  } else {
    ends.pend = null;
    ends.pendLabel = '';
  }
  $('tDayHint').textContent = `used ${q.today_used}/${q.max_per_day} today`;
  paintTimers();

  const hit = sc.hit_pct != null ? `${Number(sc.hit_pct).toFixed(0)}%` : '—';
  const oc = data.open_call;
  let priceChip = '';
  if (oc && oc.entry_price != null) {
    const pair = oc.trade_pair || '?';
    priceChip = `<span class="desk-chip border-sky-500/30 text-sky-200"><b class="font-mono">entry ${fmtPrice(pair, oc.entry_price)}</b> · mark ${fmtPrice(pair, oc.mark_price)} · ${fmtBps(oc.move_bps)}</span>`;
  }
  $('scoreline').innerHTML = `
    ${priceChip}
    <span class="desk-chip"><b class="font-mono text-emerald-400">${sc.won ?? '—'}</b> won</span>
    <span class="desk-chip"><b class="font-mono text-rose-400">${sc.lost ?? '—'}</b> lost</span>
    <span class="desk-chip"><b class="font-mono text-amber-400">${sc.washed ?? '—'}</b> wash</span>
    <span class="desk-chip"><b class="font-mono">${sc.pending ?? 0}</b> pending</span>
    <span class="desk-chip text-zinc-500">hit ${hit} · ${sc.lifetime_decisive ?? '—'} decisive · ${sc.status || '—'}${sc.meets_gate ? ' · qualified' : ''}</span>
    <span class="desk-chip ${data.submit_allowed ? 'border-emerald-500/30 text-emerald-300' : 'border-rose-500/30 text-rose-300'}">${data.submit_allowed ? 'submit open' : 'submit blocked'}</span>`;

  renderWeek('weekPanel', 'weekBadge', 'weekBody', data.week);

  renderQualifyPath(data.qualify_path);

  renderOpenCall(data.open_call);

  allCalls = data.calls || [];
  renderHistoryPage();
  if (lastMarket) renderCoinsView();
  if (submitSuccess) renderSubmitSuccessPanel();

  const nextMs = data.open_call ? 15000 : 30000;
  if (nextMs !== statusRefreshMs) {
    statusRefreshMs = nextMs;
    if (statusTimer) clearInterval(statusTimer);
    statusTimer = setInterval(refreshStatus, statusRefreshMs);
  }
}

async function refreshStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'status failed');
    renderStatus(data);
  } catch (e) {
    $('scoreline').innerHTML = `<span class="text-rose-400 text-sm">${e.message || e}</span>`;
  }
}

function headlineItem(h) {
  const tone = h.tone === 'BULL' ? 'LONG' : h.tone === 'BEAR' ? 'SHORT' : 'NONE';
  return `<li class="flex gap-2 items-start py-1.5 border-b border-zinc-800/40 last:border-0">${pill(tone)} <span class="text-zinc-400 leading-snug">${h.title}</span></li>`;
}

async function loadMarket() {
  try {
    const res = await fetch('/api/market');
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'market failed');
    lastMarket = data;
    renderCoinsView();
    if (!lastEstimate) {
      $('news').innerHTML = (data.headlines || []).slice(0, 14).map(headlineItem).join('')
        || '<li class="text-zinc-500 text-sm">No headlines</li>';
    }
  } catch (e) {
    if (!lastMarket) {
      $('coins').innerHTML = `<span class="text-rose-400 text-sm">${e.message || e}</span>`;
    }
  }
}

function submitContext() {
  const est = lastEstimate;
  const mkt = lastMarket;
  const snap = statusSnap;
  const canSub = est ? est.submit_allowed
    : (mkt ? mkt.submit_allowed : (snap ? snap.submit_allowed : false));
  const block = (est && est.submit_block_reason)
    || (mkt && mkt.submit_block_reason)
    || (snap && snap.submit_block_reason)
    || '';
  return {
    canSub: !!canSub,
    block,
    gate: Number((est && est.min_confidence) || (mkt && mkt.min_confidence) || 0),
  };
}

function expectationPlaceholder(coin) {
  const tp = coin && coin.tp_sl_bps != null ? `${coin.tp_sl_bps} bps TP/SL` : '—';
  const reach = coin && coin.reach_score != null ? `reach ${coin.reach_score}` : '';
  const atr = coin && coin.atr_to_tp_ratio != null ? `ATR/TP ${coin.atr_to_tp_ratio}` : '';
  return `<div class="text-sm text-zinc-500">Charts loaded. Click <b class="text-zinc-300">Estimate</b> for LLM direction, confidence, and thesis.</div>
    <div class="text-xs font-mono text-zinc-500 mt-2">${tp}${reach ? ' · ' + reach : ''}${atr ? ' · ' + atr : ''}</div>`;
}

function wireCoinButtons(ctx) {
  const data = lastEstimate || lastMarket;
  document.querySelectorAll('button[data-pair][data-dir]').forEach(btn => {
    btn.onclick = () => {
      const pair = btn.getAttribute('data-pair');
      const direction = btn.getAttribute('data-dir');
      const confRaw = btn.getAttribute('data-conf');
      const conf = confRaw === '' || confRaw == null ? null : Number(confRaw);
      let thesis = '';
      try { thesis = decodeURIComponent(btn.getAttribute('data-thesis') || ''); } catch (_) {}
      selectCandidate({
        trade_pair: pair,
        direction,
        confidence: conf,
        thesis,
        source: btn.classList.contains('btn-sm-good') ? 'llm_row' : 'manual_pick',
      }, data);
    };
  });
}

function renderCoinsView() {
  const market = lastMarket;
  if (!market) return;
  const est = lastEstimate;
  const hasLlm = !!(est && est.llm);
  const { canSub, block } = submitContext();
  const scoreByPair = {};
  if (hasLlm) {
    for (const s of ((est.llm && est.llm.scores) || [])) {
      scoreByPair[String(s.trade_pair || '').toUpperCase()] = s;
    }
  }

  const rows = (market.coins || []).map(c => {
    const pair = c.trade_pair;
    const s = scoreByPair[pair] || {};
    const llmDir = hasLlm ? String(s.direction || 'NONE').toUpperCase() : null;
    const conf = hasLlm && s.confidence != null ? Number(s.confidence) : null;
    const thesis = hasLlm ? (s.thesis || s.reasoning || s.tech_reason || '') : '';
    const candles = c.candles_15m_tail || [];
    const llmCell = hasLlm
      ? `${pill(llmDir)} <span class="font-mono text-xs">${conf != null ? conf.toFixed(2) : '—'}</span>`
      : '<span class="text-zinc-500 font-mono text-xs">Estimate</span>';
    const expectHtml = hasLlm ? expectationBlock(s, c) : expectationPlaceholder(c);
    return `
    <tr class="hover:bg-zinc-900/30">
      <td class="desk-td font-mono text-xs"><b class="text-white text-sm">${pair}</b><br/><span class="text-zinc-500">${c.last}</span></td>
      <td class="desk-td">${pill(c.tech_bias)} <span class="font-mono text-xs">${c.tech_score}</span></td>
      <td class="desk-td">${pill(c.news_bias)} <span class="font-mono text-xs">${c.news_score}</span></td>
      <td class="desk-td">${llmCell}</td>
      <td class="desk-td font-mono text-xs">${c.pre_score}</td>
      <td class="desk-td font-mono text-xs text-zinc-500">${c.chg_1h_pct}% / ${c.chg_4h_pct}%</td>
      <td class="desk-td whitespace-nowrap">
        <button class="btn-sm-long" ${canSub ? '' : 'disabled'}
          data-pair="${pair}" data-dir="LONG" data-conf="${conf ?? ''}"
          data-thesis="${encodeURIComponent(thesis)}" title="${canSub ? 'Select LONG' : block}">LONG</button>
        <button class="btn-sm-short ml-1" ${canSub ? '' : 'disabled'}
          data-pair="${pair}" data-dir="SHORT" data-conf="${conf ?? ''}"
          data-thesis="${encodeURIComponent(thesis)}" title="${canSub ? 'Select SHORT' : block}">SHORT</button>
      </td>
    </tr>
    <tr>
      <td colspan="7" class="desk-td !bg-zinc-950/60 !pb-5">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-3">
          <div class="rounded-xl border border-zinc-800/70 bg-zinc-950/50 p-4">
            <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">15m candles · ${candles.length} bars</div>
            ${candles.length ? candleChartBlock(pair, candles) : sparklineSvg(candles)}
            <div class="text-xs font-mono text-zinc-500 mt-2">
              1h ${c.chg_1h_pct}% · 4h ${c.chg_4h_pct}% · 24h ${c.chg_24h_pct}% · RSI ${c.rsi14} · ${c.trend || '—'}
            </div>
          </div>
          <div class="rounded-xl border border-zinc-800/70 bg-zinc-950/50 p-4">
            <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">SN89 history · ${pair}</div>
            ${coinHistoryBlock(pair)}
          </div>
          <div class="rounded-xl border border-zinc-800/70 bg-zinc-950/50 p-4">
            <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500 mb-2">${hasLlm ? 'LLM expectation' : 'Expectation'}</div>
            ${expectHtml}
          </div>
        </div>
      </td>
    </tr>`;
  }).join('');

  $('coins').innerHTML = deskTable(`<tr>
      <th class="desk-th">Pair</th><th class="desk-th">Tech</th><th class="desk-th">News</th><th class="desk-th">LLM</th><th class="desk-th">Pre</th><th class="desk-th">1h / 4h</th><th class="desk-th">Submit</th>
    </tr>`, rows);
  initCoinCharts();
  wireCoinButtons(lastEstimate || market);

  if (!hasLlm && !pending && !submitSuccess) {
    $('suggestBox').hidden = true;
  } else if (hasLlm && est.suggested && !pending && !submitSuccess) {
    selectCandidate(est.suggested, est);
  } else if (submitSuccess) {
    renderSubmitSuccessPanel();
  }
}

async function estimate() {
  $('btnEstimate').disabled = true;
  $('busy').textContent = 'Estimating…';
  try {
    const body = {
      deep: $('mode').value === 'deep',
      model: $('model').value || null,
    };
    const res = await fetch('/api/estimate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'estimate failed');
    lastEstimate = data;
    renderEstimate(data);
    $('busy').textContent = `${data.model} · $${data.budget.this_call_usd} · day $${data.budget.spent_usd}/$${data.budget.limit_usd}`;
    refreshStatus();
  } catch (e) {
    $('busy').textContent = String(e.message || e);
    $('busy').className = 'text-xs font-mono text-rose-400 ml-auto';
  } finally {
    $('btnEstimate').disabled = false;
  }
}

function renderEstimate(data) {
  lastEstimate = data;
  lastMarket = {
    ...(lastMarket || {}),
    utc_now: data.utc_now,
    submit_allowed: data.submit_allowed,
    submit_block_reason: data.submit_block_reason,
    min_confidence: data.min_confidence,
    headlines: data.headlines,
    coins: data.coins,
  };
  $('busy').className = 'text-xs font-mono text-zinc-500 ml-auto';
  $('news').innerHTML = (data.headlines || []).slice(0, 14).map(headlineItem).join('')
    || '<li class="text-zinc-500 text-sm">No headlines</li>';

  renderCoinsView();

  const canSub = !!data.submit_allowed;
  const block = data.submit_block_reason || '';
  const gate = Number(data.min_confidence || 0);
  const scores = (data.llm && data.llm.scores) || [];
  if (scores.length) {
    $('scores').innerHTML = deskTable(`<tr>
        <th class="desk-th">Pair</th><th class="desk-th">Dir</th><th class="desk-th">Conf</th><th class="desk-th">Wash</th><th class="desk-th">Why</th><th class="desk-th">Submit</th>
      </tr>`, scores.slice().sort((a,b)=>(b.confidence||0)-(a.confidence||0)).map(s => {
        const pair = String(s.trade_pair || '').toUpperCase();
        const dir = String(s.direction || 'NONE').toUpperCase();
        const conf = Number(s.confidence || 0);
        const thesis = s.thesis || s.reasoning || '';
        const enc = encodeURIComponent(thesis);
        const llmBtn = (dir === 'LONG' || dir === 'SHORT')
          ? `<button class="btn-sm-good" ${canSub ? '' : 'disabled'} data-pair="${pair}" data-dir="${dir}" data-conf="${conf}" data-thesis="${enc}">Submit ${dir}</button>`
          : `<span class="text-zinc-500 font-mono text-xs">NONE</span>`;
        return `
        <tr class="hover:bg-zinc-900/30">
          <td class="desk-td font-mono text-xs text-white">${pair}</td>
          <td class="desk-td">${pill(dir)}</td>
          <td class="desk-td font-mono text-xs">${conf.toFixed(2)}</td>
          <td class="desk-td font-mono text-xs">${s.wash_risk != null ? Number(s.wash_risk).toFixed(2) : '—'}</td>
          <td class="desk-td text-sm text-zinc-300">
            <div>${s.thesis || ''}</div>
            <details class="mt-1"><summary class="cursor-pointer text-cyan-400 text-xs font-semibold">reasoning</summary>
              <div class="whitespace-pre-wrap mt-1 text-xs text-zinc-500">${s.reasoning || '—'}</div>
            </details>
          </td>
          <td class="desk-td whitespace-nowrap">
            ${llmBtn}
            <button class="btn-sm-long ml-1" ${canSub ? '' : 'disabled'} data-pair="${pair}" data-dir="LONG" data-conf="${conf}" data-thesis="${enc}">LONG</button>
            <button class="btn-sm-short ml-1" ${canSub ? '' : 'disabled'} data-pair="${pair}" data-dir="SHORT" data-conf="${conf}" data-thesis="${enc}">SHORT</button>
          </td>
        </tr>`;
      }).join(''))
      + `<p class="text-sm text-zinc-500 mt-4 whitespace-pre-wrap leading-relaxed">${((data.llm.best)||{}).why || ''}</p>`;
    wireCoinButtons(data);
  } else {
    $('scores').innerHTML = `<pre class="text-xs font-mono text-zinc-400">${JSON.stringify(data.llm, null, 2)}</pre>`;
  }
  $('raw').textContent = JSON.stringify(data.llm, null, 2);

  if (data.suggested && !submitSuccess && !pending) {
    selectCandidate(data.suggested, data);
  } else if (submitSuccess) {
    renderSubmitSuccessPanel();
  } else if (!pending) {
    $('suggestBox').hidden = false;
    $('sugTitle').textContent = 'Pick any coin';
    $('sugMeta').textContent = canSub ? `submit open · gate ${gate}` : `Blocked: ${block}`;
    $('sugThesis').textContent = (data.llm && data.llm.best && data.llm.best.why) || 'Choose LONG or SHORT on any row.';
    $('btnSubmit').disabled = true;
    setSubStatus(canSub ? 'Select a candidate' : `Blocked: ${block}`, canSub ? '' : 'err');
  }
}

function selectCandidate(cand, data) {
  if (!cand || !cand.trade_pair || !cand.direction) return;
  const direction = String(cand.direction).toUpperCase();
  if (direction !== 'LONG' && direction !== 'SHORT') return;
  clearSubmitSuccess();
  pending = {
    trade_pair: String(cand.trade_pair).toUpperCase(),
    direction,
    confidence: cand.confidence != null ? Number(cand.confidence) : null,
    thesis: cand.thesis || '',
    source: cand.source || 'manual_pick',
  };
  const gate = submitContext().gate;
  const { canSub, block } = submitContext();
  $('suggestBox').hidden = false;
  $('sugTitle').textContent = `${pending.trade_pair} ${pending.direction}`;
  const confTxt = pending.confidence != null ? `conf ${pending.confidence.toFixed(2)}` : 'conf —';
  $('sugMeta').textContent = `${confTxt} · gate ${gate} · ${pending.source}`;
  $('sugThesis').textContent = pending.thesis || '';
  const below = pending.confidence != null && pending.confidence < gate;
  $('btnSubmit').disabled = !canSub;
  setSubStatus(
    !canSub ? `Blocked: ${block}` : (below ? 'Below conf gate — submit at your risk' : 'Ready to submit'),
    !canSub ? 'err' : (below ? 'wait' : 'ok'),
  );
}

async function submit() {
  if (!pending) return;
  if (!confirm(`Submit ${pending.trade_pair} ${pending.direction}?`)) return;
  $('btnSubmit').disabled = true;
  document.querySelectorAll('button[data-pair][data-dir]').forEach(b => { b.disabled = true; });
  setSubStatus('Submitting…', '');
  try {
    const res = await fetch('/api/submit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        trade_pair: pending.trade_pair,
        direction: pending.direction,
        thesis: pending.thesis || '',
      }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || JSON.stringify(data.result || data));
    submitSuccess = {
      pair: pending.trade_pair,
      direction: pending.direction,
      thesis: pending.thesis || '',
      at: Date.now(),
      via: data.via,
    };
    pending = null;
    renderSubmitSuccessPanel(data);
    $('raw').textContent = JSON.stringify(data, null, 2);
    await refreshStatus();
    renderSubmitSuccessPanel(data);
    if (lastEstimate) {
      if (statusSnap) {
        lastEstimate.submit_allowed = statusSnap.submit_allowed;
        lastEstimate.submit_block_reason = statusSnap.submit_block_reason;
        if (lastMarket) {
          lastMarket.submit_allowed = statusSnap.submit_allowed;
          lastMarket.submit_block_reason = statusSnap.submit_block_reason;
        }
      }
      renderEstimate(lastEstimate);
      renderSubmitSuccessPanel(data);
    } else if (lastMarket) {
      renderCoinsView();
      renderSubmitSuccessPanel(data);
    }
  } catch (e) {
    setSubStatus(String(e.message || e), 'err');
    $('btnSubmit').disabled = false;
    document.querySelectorAll('button[data-pair][data-dir]').forEach(b => { b.disabled = false; });
    if (lastEstimate) renderEstimate(lastEstimate);
    else if (lastMarket) renderCoinsView();
  }
}

$('btnEstimate').onclick = estimate;
$('btnSubmit').onclick = submit;
$('btnRefresh').onclick = () => { refreshStatus(); loadMarket(); };
$('btnClear').onclick = () => {
  pending = null;
  clearSubmitSuccess();
  $('suggestBox').hidden = true;
  $('subStatus').textContent = '';
};
$('histSearch').addEventListener('input', (e) => {
  historyState.query = e.target.value || '';
  historyState.page = 1;
  renderHistoryPage();
});
$('histDate').addEventListener('change', (e) => {
  historyState.date = e.target.value || '';
  historyState.page = 1;
  renderHistoryPage();
});
$('histPageSize').addEventListener('change', (e) => {
  historyState.pageSize = Number(e.target.value) || 10;
  historyState.page = 1;
  renderHistoryPage();
});
$('histPrev').onclick = () => {
  if (historyState.page > 1) {
    historyState.page -= 1;
    renderHistoryPage();
  }
};
$('histNext').onclick = () => {
  historyState.page += 1;
  renderHistoryPage();
};
function hfFmtBps(v) {
  if (v == null) return '—';
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  const sign = n >= 0 ? '+' : '';
  return `${sign}${n.toFixed(1)} bps`;
}

function hfDurLeft(s) {
  if (s == null) return '—';
  return fmtDur(s);
}

function fmtHfPrice(v) {
  if (v == null) return '—';
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  const dec = n >= 100 ? 2 : n >= 1 ? 4 : 6;
  return n.toFixed(dec);
}

function fmtHfHorizon(s) {
  const n = Number(s);
  if (!Number.isFinite(n) || n <= 0) return '—';
  return n >= 3600 ? `${(n / 3600).toFixed(n % 3600 ? 1 : 0)}h` : `${Math.round(n / 60)}m`;
}

const HF_CALL_STATUS = {
  won: { badge: BADGE.won, label: 'WON' },
  lost: { badge: BADGE.lost, label: 'LOST' },
  washed: { badge: BADGE.washed, label: 'WASH' },
  open: { badge: BADGE.pending, label: 'PENDING' },
  pending: { badge: BADGE.pending, label: 'PENDING' },
  refused: { badge: BADGE.void, label: 'REFUSED' },
  void: { badge: BADGE.void, label: 'VOID' },
};

function hfCallOutcomeCell(c) {
  const st = String(c.status || '').toLowerCase();
  if (st === 'won' || st === 'lost') {
    return `<span class="${bpsClass(c.outcome_bps)}">${hfFmtBps(c.outcome_bps)}</span>`;
  }
  if (st === 'washed') return '<span class="text-zinc-500">wash</span>';
  if (st === 'open' || st === 'pending') {
    const end = (Number(c.t0_unix) || 0) + (Number(c.horizon_s) || 0);
    const left = end - Date.now() / 1000;
    return left > 0
      ? `<span class="text-amber-300">${fmtDur(left)} left</span>`
      : '<span class="text-zinc-500">settling…</span>';
  }
  if (st === 'refused') {
    return `<span class="text-zinc-500 text-[11px]" title="${c.refused_reason || ''}">${c.refused_reason || '—'}</span>`;
  }
  return '—';
}

let hfCallsFilter = 'all';
let lastHfCalls = [];

function applyHfCallsFilter() {
  document.querySelectorAll('#hfCallsFilters [data-hf-filter]').forEach(btn => {
    const active = btn.dataset.hfFilter === hfCallsFilter;
    btn.classList.toggle('bg-zinc-800', active);
    btn.classList.toggle('text-white', active);
    btn.classList.toggle('border-zinc-500', active);
  });

  const filtered = hfCallsFilter === 'all' ? lastHfCalls : lastHfCalls.filter(c => {
    const st = String(c.status || '').toLowerCase();
    if (hfCallsFilter === 'pending') return st === 'open' || st === 'pending';
    return st === hfCallsFilter;
  });

  const el = $('hfCallsBody');
  if (!filtered.length) {
    el.innerHTML = `<span class="text-zinc-500 text-sm p-4 block">No ${hfCallsFilter === 'all' ? '' : hfCallsFilter + ' '}submissions yet</span>`;
    return;
  }
  const rows = filtered.map(c => {
    const st = String(c.status || '').toLowerCase();
    const tag = HF_CALL_STATUS[st] || { badge: BADGE.none, label: (c.status || '—').toUpperCase() };
    const when = c.t0_unix ? new Date(c.t0_unix * 1000).toISOString().slice(0, 19).replace('T', ' ') + ' UTC' : '—';
    return `<tr>
      <td class="desk-td font-mono text-xs text-zinc-400">${when}</td>
      <td class="desk-td font-mono text-xs text-white">${c.trade_pair || '—'}</td>
      <td class="desk-td">${pill(c.direction)}</td>
      <td class="desk-td font-mono text-xs">${fmtHfPrice(c.entry_price)}</td>
      <td class="desk-td font-mono text-xs text-zinc-400">${c.tp_bps ?? '—'} / ${c.sl_bps ?? '—'}</td>
      <td class="desk-td font-mono text-xs text-zinc-400">${fmtHfHorizon(c.horizon_s)}</td>
      <td class="desk-td"><span class="${tag.badge}">${tag.label}</span></td>
      <td class="desk-td font-mono text-xs ${bpsClass(c.move_bps)}">${hfFmtBps(c.move_bps)}</td>
      <td class="desk-td font-mono text-xs">${hfCallOutcomeCell(c)}</td>
    </tr>`;
  }).join('');
  el.innerHTML = deskTable(
    '<tr><th class="desk-th">Time (UTC)</th><th class="desk-th">Pair</th><th class="desk-th">Dir</th><th class="desk-th">Entry</th><th class="desk-th">TP/SL bps</th><th class="desk-th">Horizon</th><th class="desk-th">Status</th><th class="desk-th">Move</th><th class="desk-th">Outcome</th></tr>',
    rows,
  );
}

function renderHfCalls(calls) {
  lastHfCalls = calls || [];
  applyHfCallsFilter();
}

document.querySelectorAll('#hfCallsFilters [data-hf-filter]').forEach(btn => {
  btn.onclick = () => { hfCallsFilter = btn.dataset.hfFilter; applyHfCallsFilter(); };
});

let hfEnabled = null;
let hfMinerTag = localStorage.getItem('sn89_hf_miner') || null;

function setHfMinerTag(tag) {
  hfMinerTag = tag;
  localStorage.setItem('sn89_hf_miner', tag);
  return Promise.all([refreshHf(), refreshHfResults()]);
}

function renderHfMiners(miners, defaultTag) {
  if (!hfMinerTag || !miners.some(m => m.tag === hfMinerTag)) {
    hfMinerTag = (miners.find(m => m.tag === defaultTag) ? defaultTag : miners[0]?.tag) || defaultTag;
  }
  const el = $('hfSidebarMiners');
  if (!miners.length) {
    el.innerHTML = '<span class="text-zinc-600 text-xs px-3">no HF miners found</span>';
    return;
  }
  el.innerHTML = miners.map(m => {
    const dot = m.running
      ? (m.enabled ? 'on' : 'paused')
      : 'off';
    const meta = m.running
      ? `${m.submits_today ?? 0} today · ${m.dry_run ? 'dry-run' : 'live'}`
      : 'offline';
    const label = m.hotkey ? `${m.tag} · ${m.hotkey.slice(0, 6)}…` : m.tag;
    return `<button type="button" class="miner-item" data-miner-kind="hf" data-miner-tag="${m.tag}">
        <span class="miner-item-dot ${dot}"></span>
        <span class="miner-item-label">
          <span class="miner-item-name">${label}</span>
          <span class="miner-item-meta">${meta}</span>
        </span>
      </button>`;
  }).join('');
  el.querySelectorAll('[data-miner-tag]').forEach(btn => {
    btn.onclick = () => selectMiner('hf', btn.dataset.minerTag);
  });
  applyMinerActiveHighlight();
}

async function refreshHfMiners() {
  try {
    const res = await fetch('/api/hf/miners');
    const data = await res.json();
    if (data.ok) renderHfMiners(data.miners || [], data.default);
  } catch (e) {
    // leave whatever was last rendered — a blip here shouldn't wipe the switcher
  }
}

function renderHfPairs(candidates, rotation) {
  const el = $('hfPairs');
  if (!candidates || !candidates.length) {
    el.innerHTML = '<span class="text-zinc-500 text-sm p-4 block">Waiting for bot state…</span>';
    return;
  }
  const activePair = rotation && rotation.enabled ? rotation.active_pair : null;
  const rows = candidates.slice().sort((a, b) => (b.score || 0) - (a.score || 0)).map(c => {
    const dir = c.direction || 'NONE';
    const isActive = activePair && c.pair === activePair;
    const isIdle = activePair && !isActive;
    const pairLabel = isActive
      ? `${c.pair} <span class="desk-chip border-emerald-500/30 text-emerald-300" style="padding:0 6px;font-size:9px">today</span>`
      : c.pair;
    return `<tr class="${isIdle ? 'opacity-40' : ''}">
      <td class="desk-td font-mono text-xs text-white">${pairLabel}</td>
      <td class="desk-td font-mono text-xs">${c.price != null ? c.price : '—'}</td>
      <td class="desk-td">${pill(dir)}</td>
      <td class="desk-td font-mono text-xs">${(c.score || 0).toFixed(2)}</td>
      <td class="desk-td text-zinc-500 text-xs">${isIdle ? 'rotation: not today\'s pair' : (c.reason || '')}</td>
    </tr>`;
  }).join('');
  el.innerHTML = deskTable(
    '<tr><th class="desk-th">Pair</th><th class="desk-th">Last</th><th class="desk-th">Signal</th><th class="desk-th">Score</th><th class="desk-th">Note</th></tr>',
    rows,
  );
}

function renderHfOpen(openPositions) {
  const el = $('hfOpen');
  if (!openPositions || !openPositions.length) {
    el.innerHTML = '<span class="text-zinc-500 text-sm p-4 block">No open HF positions</span>';
    return;
  }
  const rows = openPositions.map(p => {
    // Target bps to WIN (TP, touch first = won) vs LOSE (SL, touch first =
    // lost) — fixed at submit time, same regardless of which side gets
    // touched first; shown side by side so it reads against Move directly.
    const tpSl = (p.tp_bps != null && p.sl_bps != null)
      ? `<span class="text-emerald-400">+${p.tp_bps}</span> / <span class="text-rose-400">-${p.sl_bps}</span>`
      : '—';
    return `<tr>
      <td class="desk-td font-mono text-xs text-white">${p.pair}</td>
      <td class="desk-td">${pill(p.direction)}</td>
      <td class="desk-td font-mono text-xs">${p.entry_ref != null ? p.entry_ref : '—'}</td>
      <td class="desk-td font-mono text-xs text-emerald-300/90">${p.mark_price != null ? p.mark_price : '—'}</td>
      <td class="desk-td font-mono text-xs">${tpSl}</td>
      <td class="desk-td font-mono text-xs ${bpsClass(p.move_bps)}">${hfFmtBps(p.move_bps)}</td>
      <td class="desk-td font-mono text-xs text-amber-400">${hfDurLeft(p.horizon_left_s)}</td>
    </tr>`;
  }).join('');
  el.innerHTML = deskTable(
    '<tr><th class="desk-th">Pair</th><th class="desk-th">Dir</th><th class="desk-th">Entry ref</th><th class="desk-th">Mark</th><th class="desk-th">TP/SL bps</th><th class="desk-th">Move</th><th class="desk-th">Horizon left</th></tr>',
    rows,
  );
}

function renderHfHistory(history) {
  const el = $('hfHistory');
  if (!history || !history.length) {
    el.innerHTML = '<span class="text-zinc-500 text-sm p-4 block">No decisions logged yet</span>';
    return;
  }
  const STATUS_BADGE = {
    submitted: BADGE.won, refused: BADGE.lost, blocked: BADGE.pending,
    error: BADGE.lost, dry_run: BADGE.washed,
  };
  const SOURCE_CLS = {
    manual: 'text-sky-300', failed_breakout: 'text-fuchsia-300', auto: 'text-zinc-500',
  };
  const SOURCE_LABEL = { manual: 'manual', failed_breakout: 'reversal', auto: 'auto' };
  const rows = history.slice(0, 30).map(d => {
    const when = d.at ? new Date(d.at * 1000).toISOString().slice(11, 19) + ' UTC' : '—';
    const badge = STATUS_BADGE[d.status] || BADGE.none;
    const src = d.source || 'auto';
    return `<tr>
      <td class="desk-td font-mono text-xs">${when}</td>
      <td class="desk-td font-mono text-xs text-white">${d.pair || '—'}</td>
      <td class="desk-td">${pill(d.direction)}</td>
      <td class="desk-td font-mono text-xs">${(d.score || 0).toFixed(2)}</td>
      <td class="desk-td"><span class="${badge}">${d.status || '—'}</span></td>
      <td class="desk-td font-mono text-[10px] ${SOURCE_CLS[src] || 'text-zinc-500'}">${SOURCE_LABEL[src] || src}</td>
      <td class="desk-td text-zinc-500 text-xs">${d.reason || ''}</td>
    </tr>`;
  }).join('');
  el.innerHTML = deskTable(
    '<tr><th class="desk-th">When</th><th class="desk-th">Pair</th><th class="desk-th">Dir</th><th class="desk-th">Score</th><th class="desk-th">Status</th><th class="desk-th">Source</th><th class="desk-th">Detail</th></tr>',
    rows,
  );
}

function hfSecsUntilUtcClock(hhmm) {
  // "HH:MM UTC" for a time later today (pace ETAs never span past midnight —
  // the pace gate's own math resets at the UTC day boundary) → seconds from
  // now until that clock time.
  const m = /^(\d{2}):(\d{2})/.exec(hhmm || '');
  if (!m) return null;
  const now = new Date();
  const targetSecOfDay = Number(m[1]) * 3600 + Number(m[2]) * 60;
  const nowSecOfDay = now.getUTCHours() * 3600 + now.getUTCMinutes() * 60 + now.getUTCSeconds();
  let diff = targetSecOfDay - nowSecOfDay;
  if (diff < 0) diff += 86400;
  return diff;
}

let lastHfStatus = null;
let hfDailyResetAt = 0;

function hfSubmitReadyInfo(data, pair) {
  if (!data) {
    return { ready: false, time: '—', hint: 'loading…', badge: 'checking…',
             badgeCls: 'border-zinc-700 text-zinc-500', seconds: null };
  }
  if (!data.running) {
    return { ready: false, time: 'offline', hint: 'Start sn89-hf-auto in PM2',
             badge: 'Bot offline', badgeCls: 'border-rose-500/30 text-rose-300', seconds: null };
  }
  if (data.dry_run) {
    return { ready: false, time: 'dry-run', hint: 'Bot is in dry-run — restart with --live',
             badge: 'Dry-run', badgeCls: 'border-amber-500/30 text-amber-200', seconds: null };
  }
  const cap = data.daily_cap || 30;
  const used = data.submits_today || 0;
  if (used >= cap) {
    const left = Math.max(0, hfDailyResetAt - Date.now() / 1000);
    return { ready: false, time: fmtDur(left), hint: `Daily cap ${used}/${cap} · resets UTC midnight`,
             badge: 'Cap full', badgeCls: 'border-rose-500/30 text-rose-300', seconds: left };
  }
  if (pair) {
    const open = (data.open_positions || []).find(p => p.pair === pair);
    if (open) {
      const end = open.horizon_end_unix || (Date.now() / 1000 + (open.horizon_left_s || 0));
      const left = Math.max(0, end - Date.now() / 1000);
      if (left > 0) {
        return { ready: false, time: fmtDur(left),
                 hint: `${pair} position open · wait for horizon to close`,
                 badge: 'Pair locked', badgeCls: 'border-amber-500/30 text-amber-200', seconds: left };
      }
    }
  }
  const autoNote = data.enabled ? '' : ' · auto paused, manual OK';
  return { ready: true, time: 'Ready now', hint: `Quota ${used}/${cap}${autoNote}`,
           badge: 'Ready', badgeCls: 'border-emerald-500/30 text-emerald-300', seconds: 0 };
}

function renderHfManualReady(data) {
  lastHfStatus = data;
  if (data && data.seconds_to_daily_reset != null) {
    hfDailyResetAt = Date.now() / 1000 + data.seconds_to_daily_reset;
  }
  const pair = ($('hfManualPair') && $('hfManualPair').value) || '';
  const info = hfSubmitReadyInfo(data, pair);
  const badge = $('hfManualReadyBadge');
  const timeEl = $('hfManualReadyTime');
  const hintEl = $('hfManualReadyHint');
  const btn = $('hfManualBtn');
  if (!badge || !timeEl || !hintEl) return;
  badge.textContent = info.badge;
  badge.className = `desk-chip ${info.badgeCls}`;
  timeEl.textContent = info.time;
  timeEl.className = info.ready
    ? 'text-2xl font-mono font-semibold tabular-nums mt-2 text-emerald-400'
    : (info.seconds != null
      ? 'text-2xl font-mono font-semibold tabular-nums mt-2 text-amber-200'
      : 'text-2xl font-mono font-semibold tabular-nums mt-2 text-rose-300');
  hintEl.textContent = info.hint;
  if (btn && btn.dataset.submitting !== '1') btn.disabled = !info.ready || !pair;
}

function tickHfSubmitReady() {
  if (!lastHfStatus || selectedMinerKind !== 'hf') return;
  renderHfManualReady(lastHfStatus);
  const info = hfSubmitReadyInfo(lastHfStatus, null);
  const nextEl = $('hfTNext');
  const nextHint = $('hfTNextHint');
  if (!nextEl || !nextHint) return;
  nextEl.textContent = info.time;
  nextEl.className = info.ready
    ? 'text-2xl font-mono font-semibold tabular-nums mt-2 text-emerald-400'
    : (info.seconds != null
      ? 'text-2xl font-mono font-semibold tabular-nums mt-2 text-amber-200'
      : 'text-2xl font-mono font-semibold tabular-nums mt-2 text-rose-300');
  nextHint.textContent = info.hint;
}

function renderHfTimers(data) {
  renderHfManualReady(data);
  const cap = data.daily_cap || 30;
  const used = data.submits_today || 0;
  const info = hfSubmitReadyInfo(data, null);
  const nextEl = $('hfTNext');
  const nextHint = $('hfTNextHint');
  if (nextEl && nextHint) {
    nextEl.textContent = info.time;
    nextEl.className = info.ready
      ? 'text-2xl font-mono font-semibold tabular-nums mt-2 text-emerald-400'
      : (info.seconds != null
        ? 'text-2xl font-mono font-semibold tabular-nums mt-2 text-amber-200'
        : 'text-2xl font-mono font-semibold tabular-nums mt-2 text-rose-300');
    nextHint.textContent = info.hint;
  }

  if ($('hfTDay')) $('hfTDay').textContent = data.seconds_to_daily_reset != null ? fmtDur(data.seconds_to_daily_reset) : '—';
  if ($('hfTDayHint')) $('hfTDayHint').textContent = `used ${used}/${cap} today`;

  const pend = (data.open_positions || [])
    .slice()
    .sort((a, b) => (a.horizon_left_s ?? Infinity) - (b.horizon_left_s ?? Infinity))[0];
  const pendEl = $('hfTPend');
  const pendHint = $('hfTPendHint');
  const priceBox = $('hfTPriceBox');
  const moveLine = $('hfTMoveLine');
  if (!pendEl || !pendHint) return;
  if (!pend) {
    pendEl.textContent = 'none';
    pendHint.textContent = 'no open call';
    if (priceBox) priceBox.style.display = 'none';
    if (moveLine) moveLine.style.display = 'none';
    return;
  }
  pendEl.textContent = fmtDur(pend.horizon_left_s || 0);
  pendHint.textContent = `${pend.pair || '?'} ${String(pend.direction || '').toUpperCase()}`;
  if (pend.entry_ref != null && pend.mark_price != null) {
    if (priceBox) priceBox.style.display = 'grid';
    if ($('hfTEntry')) $('hfTEntry').textContent = fmtPrice(pend.pair, pend.entry_ref);
    if ($('hfTMark')) $('hfTMark').textContent = fmtPrice(pend.pair, pend.mark_price);
    if (moveLine) {
      moveLine.style.display = 'block';
      moveLine.innerHTML = pend.move_bps != null
        ? `<span class="${bpsClass(pend.move_bps)}">${fmtBps(pend.move_bps)}</span> vs entry`
        : '';
    }
  } else {
    if (priceBox) priceBox.style.display = 'none';
    if (moveLine) moveLine.style.display = 'none';
  }
}

function renderHfStatus(data) {
  hfEnabled = !!data.enabled;
  renderHfTimers(data);
  const runBadge = $('hfRunBadge');
  if (!data.running) {
    runBadge.textContent = 'bot not running';
    runBadge.className = 'desk-chip border-rose-500/30 text-rose-300';
  } else {
    runBadge.textContent = data.enabled ? (data.dry_run ? 'running · dry-run' : 'running · live') : 'paused';
    runBadge.className = data.enabled
      ? 'desk-chip border-emerald-500/30 text-emerald-300'
      : 'desk-chip border-amber-500/30 text-amber-200';
  }
  $('hfQuotaBadge').textContent = `${data.submits_today || 0}/${data.daily_cap || 30} today`;
  const resetBadge = $('hfResetBadge');
  if (data.seconds_to_daily_reset != null) {
    resetBadge.textContent = `resets in ${fmtDur(data.seconds_to_daily_reset)}`;
    resetBadge.className = 'desk-chip border-zinc-700 text-zinc-400';
  } else {
    resetBadge.style.display = 'none';
  }
  const pace = data.pace;
  const paceBadge = $('hfPaceBadge');
  if (pace && pace.enabled) {
    paceBadge.style.display = '';
    if (pace.on_pace) {
      paceBadge.textContent = `on pace (budget ${pace.budgeted_by_now ?? '—'})`;
      paceBadge.className = 'desk-chip border-emerald-500/30 text-emerald-300';
    } else if (pace.cap_exhausted) {
      paceBadge.textContent = 'cap exhausted \u2014 resets 00:00 UTC';
      paceBadge.className = 'desk-chip border-rose-500/30 text-rose-300';
    } else {
      paceBadge.textContent = `paced \u2014 next slot ~${pace.next_slot_eta || '—'}`;
      paceBadge.className = 'desk-chip border-amber-500/30 text-amber-200';
    }
  } else {
    paceBadge.style.display = 'none';
  }
  const rotation = data.rotation;
  const rotationBadge = $('hfRotationBadge');
  if (rotation && rotation.enabled) {
    rotationBadge.style.display = '';
    rotationBadge.textContent = `today: ${rotation.active_pair || '—'} (rotation)`;
    rotationBadge.className = 'desk-chip border-sky-500/30 text-sky-300';
  } else {
    rotationBadge.style.display = 'none';
  }
  const btn = $('hfToggleBtn');
  btn.textContent = hfEnabled ? 'Pause' : 'Resume';
  btn.disabled = false;

  const pairSel = $('hfManualPair');
  const boardPairs = Object.keys(data.board || {}).sort();
  const prevPick = pairSel.value;
  if (pairSel.dataset.pairs !== boardPairs.join(',')) {
    pairSel.dataset.pairs = boardPairs.join(',');
    pairSel.innerHTML = boardPairs.map(p => `<option value="${p}">${p}</option>`).join('')
      || '<option value="">no pairs</option>';
    if (boardPairs.includes(prevPick)) pairSel.value = prevPick;
  }

  renderHfPairs(data.candidates, rotation);
  renderHfOpen(data.open_positions);
  renderHfHistory(data.history);
}

async function refreshHf() {
  try {
    const res = await fetch(`/api/hf/status?miner=${encodeURIComponent(hfMinerTag || '')}`);
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'hf status failed');
    renderHfStatus(data);
  } catch (e) {
    $('hfRunBadge').textContent = 'unavailable';
    $('hfRunBadge').className = 'desk-chip border-zinc-700 text-zinc-500';
  }
}

function renderHfResults(data) {
  const hk = $('hfResultsHotkey');
  hk.textContent = data && data.hotkey ? `${data.hotkey.slice(0, 8)}…${data.hotkey.slice(-4)}` : '—';
  if (!data || !data.ok) {
    $('hfResultsStatusBadge').textContent = 'unavailable';
    $('hfResultsStatusBadge').className = 'desk-chip border-zinc-700 text-zinc-500';
    $('hfResultsEligBadge').textContent = '—';
    $('hfResultsEligBadge').className = 'desk-chip border-zinc-700 text-zinc-500';
    $('hfResultsQualBadge').textContent = '—';
    $('hfResultsQualBadge').className = 'desk-chip border-zinc-700 text-zinc-500';
    $('hfResultsBody').innerHTML = `<span class="text-zinc-500">${(data && data.error) || 'Could not load HF results from the IQ API.'}</span>`;
    renderWeek('hfWeekPanel', 'hfWeekBadge', 'hfWeekBody', null);
    renderHfCalls([]);
    return;
  }

  renderWeek('hfWeekPanel', 'hfWeekBadge', 'hfWeekBody', data.week);

  $('hfResultsStatusBadge').textContent = data.status || '—';
  $('hfResultsStatusBadge').className = data.status === 'active'
    ? 'desk-chip border-emerald-500/40 text-emerald-300 bg-emerald-500/10'
    : 'desk-chip border-amber-500/40 text-amber-200 bg-amber-500/10';
  // "Eligible" = volume gate (submissions + trading days) — earns nothing
  // until this flips, regardless of hit rate. "Qualified" = the Wilson LB
  // confidence gate on top, which decides whether wins post-eligibility
  // actually price above zero. They are independent and both required.
  $('hfResultsEligBadge').textContent = data.eligible ? 'Eligible' : 'Not eligible yet';
  $('hfResultsEligBadge').className = data.eligible
    ? 'desk-chip border-emerald-500/40 text-emerald-300 bg-emerald-500/10'
    : 'desk-chip border-amber-500/40 text-amber-200 bg-amber-500/10';
  $('hfResultsQualBadge').textContent = data.qualified ? 'Qualified' : 'Not qualified yet';
  $('hfResultsQualBadge').className = data.qualified
    ? 'desk-chip border-emerald-500/40 text-emerald-300 bg-emerald-500/10'
    : 'desk-chip border-amber-500/40 text-amber-200 bg-amber-500/10';

  const statTile = (label, value, cls) => `<div class="rounded-xl border border-zinc-800/60 px-3 py-2.5">
    <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500">${label}</div>
    <div class="font-mono text-lg mt-1 ${cls || 'text-zinc-200'}">${value}</div>
  </div>`;
  const decisive = data.won + data.lost;
  const hitPct = data.hit_rate_pct != null ? data.hit_rate_pct : (decisive ? (data.won / decisive * 100) : 0);
  // IQ conf_hit_pct = shrunk hit-rate (tier metric). Wilson LB is the real
  // qualify gate (≥50% with ≥8 decisive) — computed locally from won/lost.
  const shrunkPct = data.conf_hit_pct;
  const wilsonPct = data.wilson_lb_pct;
  const wilsonOk = !!data.wilson_lb_ok;
  const emissionWeight = data.emission_weight;
  const tiles = [
    statTile('Won', data.won, 'text-emerald-300'),
    statTile('Lost', data.lost, 'text-rose-300'),
    statTile('Washed', data.washed, 'text-zinc-400'),
    statTile('Pending', data.pending, 'text-amber-300'),
    statTile('Hit rate', `${Number(hitPct).toFixed(1)}%`, hitPct >= 50 ? 'text-emerald-300' : 'text-amber-300'),
    statTile('Wilson LB', wilsonPct != null ? `${Number(wilsonPct).toFixed(1)}%` : '—',
      wilsonOk ? 'text-emerald-300' : 'text-amber-300'),
    statTile('Shrunk hit (IQ)', shrunkPct != null ? `${Number(shrunkPct).toFixed(1)}%` : '—',
      (shrunkPct || 0) >= 50 ? 'text-zinc-300' : 'text-zinc-500'),
    statTile('Avg R', data.avg_r != null ? data.avg_r.toFixed(2) : '—', (data.avg_r || 0) >= 0 ? 'text-emerald-300' : 'text-rose-300'),
    statTile('Emission weight', emissionWeight != null ? emissionWeight.toFixed(4) : '—', (emissionWeight || 0) > 0 ? 'text-emerald-300' : 'text-zinc-400'),
  ].join('');

  const g = data.gate || {};
  const dv = g.diversity || {};
  const subPct = g.submissions_required ? Math.min(100, g.submissions / g.submissions_required * 100) : 0;
  const dayPct = g.trading_days_required ? Math.min(100, g.trading_days / g.trading_days_required * 100) : 0;
  const bar = (pct, ok) => `<div class="h-1.5 rounded-full bg-zinc-800 overflow-hidden mt-1.5"><div class="h-full ${ok ? 'bg-emerald-400' : 'bg-amber-400'}" style="width:${Math.max(2, pct)}%"></div></div>`;

  const eligibility = `
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
      <div class="rounded-xl border border-zinc-800/60 px-3 py-2.5">
        <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Accepted submissions</div>
        <div class="font-mono text-sm mt-1 text-zinc-200">${g.submissions ?? '—'} / ${g.submissions_required ?? '—'}
          ${g.submissions_remaining ? `<span class="text-zinc-500">· ${g.submissions_remaining} to go</span>` : ''}</div>
        ${bar(subPct, subPct >= 100)}
      </div>
      <div class="rounded-xl border border-zinc-800/60 px-3 py-2.5">
        <div class="text-[10px] font-bold uppercase tracking-wider text-zinc-500">Distinct trading days</div>
        <div class="font-mono text-sm mt-1 text-zinc-200">${g.trading_days ?? '—'} / ${g.trading_days_required ?? '—'}
          ${g.trading_days_remaining ? `<span class="text-zinc-500">· ${g.trading_days_remaining} to go</span>` : ''}</div>
        ${bar(dayPct, dayPct >= 100)}
      </div>
    </div>`;

  const diversityNote = dv.applies
    ? `Minority-side share ${((dv.share || 0) * 100).toFixed(1)}% vs floor ${((dv.floor || 0) * 100).toFixed(0)}% — ${dv.ok ? 'passing' : 'FAILING'}.`
    : `Not applied yet at this sample size (minority-side share ${((dv.share || 0) * 100).toFixed(1)}%).`;

  const byPair = dv.by_pair || {};
  const assetRows = (data.assets || []).map(a => {
    const won = a.won || 0, lost = a.lost || 0, washed = a.washed || 0;
    const decisive = won + lost;
    const hp = decisive ? (won / decisive * 100).toFixed(1) : '—';
    // a.n is the pair's TOTAL submissions including still-open ones — won +
    // lost + washed only covers RESOLVED calls, so show pending explicitly
    // rather than leaving N looking like it doesn't add up.
    const pending = Math.max(0, (a.n || 0) - won - lost - washed);
    return `<tr>
      <td class="desk-td font-mono text-xs text-white">${a.asset}</td>
      <td class="desk-td font-mono text-xs">${a.n}</td>
      <td class="desk-td font-mono text-xs text-emerald-300">${won}</td>
      <td class="desk-td font-mono text-xs text-rose-300">${lost}</td>
      <td class="desk-td font-mono text-xs text-zinc-400">${washed}</td>
      <td class="desk-td font-mono text-xs text-amber-300">${pending}</td>
      <td class="desk-td font-mono text-xs">${hp}${hp !== '—' ? '%' : ''}</td>
    </tr>`;
  }).join('');

  const streak = data.streak;
  const streakNote = streak && streak.n
    ? `Current streak: <b class="${streak.kind === 'W' ? 'text-emerald-300' : 'text-rose-300'}">${streak.n} ${streak.kind === 'W' ? 'win' : 'loss'}${streak.n === 1 ? '' : (streak.kind === 'W' ? 'wins' : 'es')}</b> in a row.`
    : '';

  $('hfResultsBody').innerHTML = `
    <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-4">${tiles}</div>
    ${eligibility}
    <p class="text-xs text-zinc-500 mb-4 leading-relaxed">
      <b class="text-zinc-400">Eligible</b> = volume gate above (submissions + trading days) — earns nothing until true.
      <b class="text-zinc-400">Qualified</b> = eligible AND <b class="text-zinc-300">Wilson LB ≥ 50%</b>
      (with ≥8 decisive). Wilson LB is the validator gate; IQ's shrunk hit is only for tiers.
      Diversity gate (${byPair && Object.keys(byPair).length || 0} pair${Object.keys(byPair).length === 1 ? '' : 's'} seen): ${diversityNote}
      ${streakNote ? ' ' + streakNote : ''}
    </p>
    ${deskTable(
      '<tr><th class="desk-th">Pair</th><th class="desk-th">N</th><th class="desk-th">Won</th><th class="desk-th">Lost</th><th class="desk-th">Wash</th><th class="desk-th">Pending</th><th class="desk-th">Hit</th></tr>',
      assetRows || '<tr><td class="desk-td text-zinc-500" colspan="7">No graded calls yet</td></tr>',
    )}
    <p class="text-[10px] text-zinc-600 mt-3 font-mono">Snapshot ${data.snapshot_at || '—'} · won/lost/wash graded server-side on the anchored tick feed</p>`;
  renderHfCalls(data.calls);
}

async function refreshHfResults() {
  try {
    const res = await fetch(`/api/hf/results?miner=${encodeURIComponent(hfMinerTag || '')}`);
    renderHfResults(await res.json());
  } catch (e) {
    renderHfResults(null);
  }
}

$('hfToggleBtn').onclick = async () => {
  const btn = $('hfToggleBtn');
  btn.disabled = true;
  try {
    const res = await fetch('/api/hf/toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: !hfEnabled, miner: hfMinerTag }),
    });
    const data = await res.json();
    if (data.ok) hfEnabled = data.enabled;
  } finally {
    refreshHf();
    refreshHfMiners();
  }
};

$('hfManualBtn').onclick = async () => {
  const pair = $('hfManualPair').value;
  const direction = $('hfManualDir').value;
  const statusEl = $('hfManualStatus');
  if (!pair) { statusEl.textContent = 'no live pairs to pick from'; statusEl.className = 'font-mono text-xs text-rose-300'; return; }
  // Fans out to EVERY discovered miner, not just the selected tab (per user
  // request 2026-08-31: "if I submit manually, submit both").
  if (!confirm(`Manually submit ${pair} ${direction} on ALL HF miners? This bypasses scoring/streak/pacing but still runs through the rate/pair-open/cross-lock gates on each one independently.`)) return;
  const btn = $('hfManualBtn');
  btn.dataset.submitting = '1';
  btn.disabled = true;
  statusEl.textContent = 'Submitting…';
  statusEl.className = 'font-mono text-xs text-zinc-400';
  try {
    const res = await fetch('/api/hf/manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pair, direction }),
    });
    const data = await res.json();
    const results = data.results || {};
    const parts = Object.entries(results).map(([tag, r]) =>
      `${tag}: ${r.ok ? (r.dry_run ? 'queued (dry-run)' : 'queued') : `failed (${r.error})`}`);
    if (!data.ok && !parts.length) throw new Error(data.error || 'manual submit failed');
    statusEl.textContent = parts.join(' · ') || 'Queued';
    statusEl.className = Object.values(results).every(r => r.ok)
      ? 'font-mono text-xs text-emerald-300' : 'font-mono text-xs text-amber-300';
    setTimeout(() => { refreshHf(); refreshHfResults(); }, 4000);
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    statusEl.className = 'font-mono text-xs text-rose-300';
  } finally {
    btn.dataset.submitting = '0';
    if (lastHfStatus) renderHfManualReady(lastHfStatus);
    else btn.disabled = false;
  }
};

$('hfManualPair').addEventListener('change', () => {
  if (lastHfStatus) renderHfManualReady(lastHfStatus);
});
$('hfManualDir').addEventListener('change', () => {
  if (lastHfStatus) renderHfManualReady(lastHfStatus);
});

setInterval(paintTimers, 1000);
statusTimer = setInterval(refreshStatus, statusRefreshMs);
setInterval(loadMarket, 120000);
setInterval(refreshHf, 5000);
// Results hit the external IQ status API (not the local bot state), so this
// polls much slower than refreshHf — grading/eligibility don't change every
// few seconds anyway.
setInterval(refreshHfResults, 30000);
// The miner list itself (which tags exist, running/enabled dots) changes
// rarely — polled at the same slow cadence as results.
setInterval(refreshHfMiners, 30000);
refreshStatus();
loadMarket();
refreshHfMiners().then(() => { refreshHf(); refreshHfResults(); });

// ── layout: left sidebar picks the miner (LF, or one of the HF tags),
// right panel shows that miner's Summary / Open Positions / Live Signal /
// History sub-tab. Background refreshes above keep running regardless of
// which miner/sub-tab is on screen.

function applyMinerActiveHighlight() {
  document.querySelectorAll('.miner-item').forEach((el) => {
    const kind = el.dataset.minerKind;
    const tag = el.dataset.minerTag || '';
    const active = kind === selectedMinerKind && (kind === 'lf' || tag === selectedMinerTagUI);
    el.classList.toggle('active', active);
  });
}

function setSubTab(subtab) {
  if (!SUBTAB_IDS.includes(subtab)) subtab = 'summary';
  document.querySelectorAll('[data-subtab-panel]').forEach((el) => {
    el.hidden = el.dataset.subtabPanel !== subtab;
  });
  document.querySelectorAll('#subTabNav .desk-tab').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.subtab === subtab);
  });
  try { localStorage.setItem(SUBTAB_KEY, subtab); } catch (_e) {}
}

function showRightPanelLoading(show) {
  const overlay = $('rightPanelLoading');
  if (!overlay) return;
  overlay.hidden = !show;
  // Belt-and-suspenders: also clear inline display in case a prior style
  // stickied display:flex over the [hidden] rule.
  if (!show) overlay.style.display = 'none';
  else overlay.style.display = '';
}

async function selectMiner(kind, tag) {
  const newKind = kind === 'hf' ? 'hf' : 'lf';
  const newTag = newKind === 'hf' ? (tag || '') : '';
  const changed = newKind !== selectedMinerKind || newTag !== selectedMinerTagUI;
  selectedMinerKind = newKind;
  selectedMinerTagUI = newTag;
  document.querySelectorAll('[data-miner-view]').forEach((el) => {
    el.hidden = el.dataset.minerView !== selectedMinerKind;
  });
  applyMinerActiveHighlight();
  try {
    localStorage.setItem(MINER_KEY, JSON.stringify({ kind: selectedMinerKind, tag: selectedMinerTagUI }));
  } catch (_e) {}

  if (!changed) {
    showRightPanelLoading(false);
    return;
  }
  const token = ++rightPanelLoadToken;
  showRightPanelLoading(true);
  try {
    if (newKind === 'hf' && newTag) {
      await setHfMinerTag(newTag);
    } else if (newKind === 'lf') {
      await refreshStatus();
    }
  } catch (_e) {
    // a failed refresh shouldn't leave the panel stuck under the overlay —
    // whatever was last rendered stays on screen, same as the background pollers.
  } finally {
    // Only the latest switch may clear the overlay — an older in-flight
    // selectMiner finishing after a newer click must not hide mid-load.
    if (token === rightPanelLoadToken) showRightPanelLoading(false);
  }
}

$('lfSidebarItem').onclick = () => selectMiner('lf', '');
document.querySelectorAll('#subTabNav .desk-tab').forEach((btn) => {
  btn.onclick = () => setSubTab(btn.dataset.subtab);
});

let initialSubtab = 'summary';
try { initialSubtab = localStorage.getItem(SUBTAB_KEY) || 'summary'; } catch (_e) {}
setSubTab(initialSubtab);

let initialMinerKind = 'lf';
let initialMinerTag = '';
try {
  const saved = JSON.parse(localStorage.getItem(MINER_KEY) || 'null');
  if (saved && saved.kind) { initialMinerKind = saved.kind; initialMinerTag = saved.tag || ''; }
} catch (_e) {}
selectMiner(initialMinerKind, initialMinerTag);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            self._send(200, HTML.encode(), "text/html; charset=utf-8")
            return
        if path == "/api/health":
            state = bot.load_state()
            submit_ok, block_reason = bot.can_submit(state)
            serve = _serve_probe()
            self._json(200, {
                "ok": True,
                "serve_url": bot.SERVE_URL,
                "serve": serve,
                "submit_mode": bot.SUBMIT_MODE,
                "wallet": f"{bot.WALLET_NAME}/{bot.WALLET_HOTKEY}",
                "hotkey": bot._resolve_hotkey_ss58(),
                "submit_allowed": submit_ok,
                "submit_block_reason": None if submit_ok else block_reason,
                "model": bot.OPENAI_MODEL,
                "has_key": bool(bot.OPENAI_API_KEY),
            })
            return
        if path == "/api/status":
            return self._json(200, run_status())
        if path == "/api/market":
            return self._json(200, run_market())
        if path == "/api/hf/miners":
            return self._json(200, run_hf_miners())
        if path == "/api/hf/status":
            tag = (qs.get("miner") or [DEFAULT_HF_TAG])[0] or DEFAULT_HF_TAG
            return self._json(200, run_hf_status(tag))
        if path == "/api/hf/results":
            tag = (qs.get("miner") or [DEFAULT_HF_TAG])[0] or DEFAULT_HF_TAG
            return self._json(200, run_hf_results(tag))
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode() or "{}")
        except Exception:
            return self._json(400, {"ok": False, "error": "bad json"})

        try:
            if path == "/api/estimate":
                if not bot.OPENAI_API_KEY:
                    return self._json(400, {"ok": False, "error": "OPENAI_API_KEY missing in bots/.env"})
                model = payload.get("model") or None
                deep = bool(payload.get("deep", True))
                return self._json(200, run_estimate(model=model, deep=deep))
            if path == "/api/submit":
                return self._json(200, run_submit(
                    str(payload.get("trade_pair") or ""),
                    str(payload.get("direction") or ""),
                    str(payload.get("thesis") or ""),
                ))
            if path == "/api/hf/toggle":
                return self._json(200, run_hf_toggle(
                    bool(payload.get("enabled", True)),
                    str(payload.get("miner") or DEFAULT_HF_TAG),
                ))
            if path == "/api/hf/manual":
                # Per user request 2026-08-31 ("if I submit manually, submit
                # both"): always fans out to every discovered miner instead
                # of only the one currently selected in the tab switcher.
                return self._json(200, run_hf_manual_submit_all(
                    str(payload.get("pair") or ""),
                    str(payload.get("direction") or ""),
                ))
            return self._json(404, {"ok": False, "error": "not found"})
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")[:800]
            return self._json(502, {"ok": False, "error": f"upstream HTTP {e.code}: {err}"})
        except Exception as e:
            return self._json(500, {
                "ok": False,
                "error": str(e),
                "trace": traceback.format_exc()[-1500:],
            })

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> int:
    p = argparse.ArgumentParser(description="SN89 AI dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    if not bot.OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not loaded — Estimate will fail until bots/.env is set")
    print(f"SN89 dashboard  http://{args.host}:{args.port}")
    print(f"  model={bot.OPENAI_MODEL}  serve={bot.SERVE_URL}  mode={bot.SUBMIT_MODE}")
    print(f"  wallet={bot.WALLET_NAME}/{bot.WALLET_HOTKEY}  direct→{bot.MINER_ROOT}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
