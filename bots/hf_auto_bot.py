#!/usr/bin/env python3
"""SN89 HF auto-bot — tick-native automated submissions for the HIGH-FREQUENCY
mechanism (mecid 1, `sn89_signals/hf.py`).

Why this is a separate bot from ai_finance_bot.py (LF): HF calls resolve in
30-120 minutes with a 250ms min submit gap and up to 30 calls/day. An LLM round
trip (seconds, plus per-call cost) cannot live in that loop, so this bot runs a
fast deterministic technical rule engine straight off live ticks instead.

Architecture:
  1) One reader thread per HF board pair holds an SSE connection open to the
     same live-quote stream `neurons.miner.cmd_limit` rests limit orders on
     (`GET {PARTNER_API}/api/sn89/closers/stream?pair=X`), feeding a rolling
     in-memory tick buffer per pair.
  2) A single decision loop (not one thread per pair — keeps the shared
     rate-limit/open-position state race-free) scores every pair's buffer with
     a tick-native momentum + mean-reversion + volatility-vs-band strategy and
     picks the best candidate above threshold.
  3) Before ever calling submit_hf(), a LOCAL safety gate mirrors (never
     replaces — the ingest is authoritative) the HF consensus rules: daily cap
     + 250ms min gap (`hf.check_rate`), one-open-position-per-pair
     (`hf.check_pair_open` / `hf.open_until_ms`, fed by our own tick buffer),
     and the 24h cross-mechanism pair lock shared with LF (checked against the
     public miner API's recent LF call history, since the local LF submit log
     does not carry per-pair info).
  4) On a pass, calls `neurons.miner.submit_hf()` in-process (no subprocess —
     latency matters here) and logs the receipt/refusal to a small state file
     the dashboard reads (`bots/dashboard.py`'s /api/hf/status).

Setup:
  # terminal 1 (only needed once per box for identity/state, no serve required)
  python bots/hf_auto_bot.py --once --dry-run
  # background, real submissions:
  python bots/hf_auto_bot.py --live

Not financial advice. Tune the SN89_HF_AUTO_* env knobs — this is a template
strategy, not a guaranteed edge.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sn89_signals import hf  # noqa: E402


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
_load_dotenv(Path.home() / ".sn89" / "hf_auto_bot.env")

# ── config ────────────────────────────────────────────────────────────────
PARTNER_API = os.getenv("SN89_PARTNER_API", "https://partner.infinitequant.app")
WALLET_NAME = os.getenv("WALLET_NAME", "GOLD")
WALLET_HOTKEY = os.getenv("WALLET_HOTKEY", "iq89")

SN89_DIR = Path.home() / ".sn89"


def _sanitize_tag(tag: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in tag) or "default"


# INSTANCE_TAG identifies WHICH miner's state/enable-flag files this process
# reads/writes — needed once more than one hf_auto_bot instance can run on the
# same box (multiple hotkeys). Explicit SN89_HF_AUTO_TAG wins so the dashboard
# (a DIFFERENT process, with its own unrelated WALLET_HOTKEY=GOLD/iq89 for LF)
# can still name any bot's files precisely via state_path_for/enable_flag_path_for
# below, rather than only ever resolving to its OWN ambient WALLET_HOTKEY.
INSTANCE_TAG = _sanitize_tag(os.getenv("SN89_HF_AUTO_TAG", "").strip() or WALLET_HOTKEY)


def state_path_for(tag: str) -> Path:
    return SN89_DIR / f"hf_auto_state_{_sanitize_tag(tag)}.json"


def enable_flag_path_for(tag: str) -> Path:
    return SN89_DIR / f"hf_auto_enabled_{_sanitize_tag(tag)}"


def manual_request_path_for(tag: str) -> Path:
    """Dashboard writes {"pair", "direction", "requested_at"} here to ask a
    RUNNING bot to submit right now, bypassing scoring/streak/rotation — this
    is a one-shot mailbox, not a queue (cycle() reads-then-deletes it)."""
    return SN89_DIR / f"hf_auto_manual_{_sanitize_tag(tag)}.json"


STATE_PATH = state_path_for(INSTANCE_TAG)
ENABLE_FLAG_PATH = enable_flag_path_for(INSTANCE_TAG)
MANUAL_REQUEST_PATH = manual_request_path_for(INSTANCE_TAG)


def _submit_log_path(hk: str) -> Path:
    return SN89_DIR / f"hf_auto_submits_{hk}.json"


def _receipts_path(hk: str) -> Path:
    return SN89_DIR / f"hf_receipts_{hk}.jsonl"


def _entry_cache_path(hk: str) -> Path:
    return SN89_DIR / f"hf_auto_entry_cache_{hk}.json"


# Pairs to trade — default is the whole live HF board; narrow with a
# comma-separated allowlist (e.g. "BTCUSD,ETHUSD,XAUUSD").
PAIR_ALLOWLIST = {
    p.strip().upper() for p in os.getenv("SN89_HF_AUTO_PAIRS", "").split(",") if p.strip()
}
EVAL_INTERVAL_S = float(os.getenv("SN89_HF_AUTO_EVAL_INTERVAL_S", "1.5"))
BUFFER_WINDOW_S = float(os.getenv("SN89_HF_AUTO_BUFFER_WINDOW_S", "3600"))  # 60 min
MIN_TICKS = int(os.getenv("SN89_HF_AUTO_MIN_TICKS", "40"))
# Calibrated for the 1800s (30 min) board rows; evaluate_pair scales these by
# horizon_s / REF_HORIZON_S per pair, so a 7200s (2h) pair like TAOUSD reads
# momentum over proportionally wider windows instead of a clock 4x too fast
# for its own resolve time.
MOM_FAST_S = float(os.getenv("SN89_HF_AUTO_MOM_FAST_S", "60"))
MOM_SLOW_S = float(os.getenv("SN89_HF_AUTO_MOM_SLOW_S", "300"))
REF_HORIZON_S = 1800.0
MIN_SCORE = float(os.getenv("SN89_HF_AUTO_MIN_SCORE", "0.28"))
# Minimum long_s/short_s dominance to call a direction at all. MUST stay well
# below MIN_VOL_RATIO: long_s/short_s are built from mom_*_bps/tp_bps, and
# |mom_*_bps| <= realized_range_bps always (a window's net move can't exceed
# the series' own high-low), so long_s <= vol_ratio unconditionally. If this
# threshold were >= MIN_VOL_RATIO, "too_quiet" could never fire on anything
# that also cleared the direction edge — the two gates would silently overlap
# to 100%, leaving the quiet gate dead code. Keeping this smaller is what
# leaves a real band (MIN_EDGE..MIN_VOL_RATIO) where a technically-directional
# but still-too-quiet drift gets caught.
MIN_EDGE = float(os.getenv("SN89_HF_AUTO_MIN_EDGE", "0.06"))
# Realized range (bps) vs the pair's own band: too far below and a touch is
# unlikely before the horizon washes; skip the pair rather than force a call.
MIN_VOL_RATIO = float(os.getenv("SN89_HF_AUTO_MIN_VOL_RATIO", "0.15"))
# If the slow-window move already covers this fraction of the band, the easy
# room to TP is gone — same "extension vs TP" idea as ai_finance_bot's LF edge.
MAX_EXTENSION_RATIO = float(os.getenv("SN89_HF_AUTO_MAX_EXTENSION_RATIO", "0.65"))
# A candidate must hold the SAME direction for this many consecutive eval
# cycles before it is actionable (gated/submitted) — cheap noise filter against
# a single reverting tick flipping the pick, at the cost of ~CONFIRM_CYCLES *
# EVAL_INTERVAL_S of reaction latency (default ~3s). Purely a submission-time
# filter: every raw candidate still lands in the state file for visibility.
CONFIRM_CYCLES = int(os.getenv("SN89_HF_AUTO_CONFIRM_CYCLES", "2"))

# ── daily-budget pacing (user request 2026-08-31) ───────────────────────────
# Without this, every pair independently races for the SAME shared 30/day cap
# and a single choppy hour can exhaust it entirely — then the bot goes
# completely dark for the rest of the UTC day even though 22+ hours of market
# still remain. This spreads the cap pro-rata across the UTC day instead.
PACE_ENABLED = os.getenv("SN89_HF_AUTO_PACE", "1").strip().lower() not in ("0", "false", "no")
# Allowance ahead of the pro-rated line — lets the bot use a few submissions
# right after each UTC-midnight reset (or right after starting mid-day)
# without waiting for the clock to "catch up", while still capping how far
# ahead of schedule it can get (which is exactly what caused the 30-in-90min
# burn this was built to prevent).
PACE_SLACK = float(os.getenv("SN89_HF_AUTO_PACE_SLACK", "2"))

# ── aggressive-spike fade (user request 2026-08-31) ─────────────────────────
# A fast, concentrated burst (stop-hunt / thin-book air-pocket) tends to give
# back part of the move before a 30min-2h horizon resolves, rather than keep
# running at that pace — so evaluate_pair treats a genuinely ACCELERATING move
# as a preferred FADE setup, not just something to avoid chasing. Fixed, NOT
# horizon-scaled like MOM_FAST_S/MOM_SLOW_S: this is about order-flow velocity
# in wall-clock time, not the trade's own holding period.
SPIKE_WINDOW_S = float(os.getenv("SN89_HF_AUTO_SPIKE_WINDOW_S", "15"))
# Band-relative floor: the burst must cover at least this fraction of the
# pair's own TP band within SPIKE_WINDOW_S, or a tiny wiggle on a near-dead
# book could clear the acceleration ratio below on baseline velocity alone.
SPIKE_MIN_RATIO = float(os.getenv("SN89_HF_AUTO_SPIKE_MIN_RATIO", "0.25"))
# The KEY distinction from ordinary momentum: how many multiples of the
# established mom_fast velocity the most-recent SPIKE_WINDOW_S burst must hit
# to count as "aggressive" rather than just the tail of an already-smooth,
# constant-rate trend (which reads recent velocity ~= fast velocity, ratio
# ~= 1, and is deliberately NOT faded — see test_sustained_uptrend/downtrend).
SPIKE_ACCEL_RATIO = float(os.getenv("SN89_HF_AUTO_SPIKE_ACCEL_RATIO", "2.5"))
# Chase-direction damping when a spike fires: the move that just happened
# gets DISCOUNTED (not just left alone), because momentum's own fast/slow
# windows unavoidably contain the same burst and would otherwise still chase
# it. Deliberately stronger than the mean-reversion guard's 0.5x (below) —
# velocity-based spike detection is a more specific exhaustion signal than
# mere range position.
SPIKE_CHASE_DAMPEN = float(os.getenv("SN89_HF_AUTO_SPIKE_CHASE_DAMPEN", "0.3"))
# Fade-direction score floor once a spike fires — "prefer", not "always win":
# max()'d against whatever the fade side already scored from momentum, and
# scaled by how big the spike is (bigger burst -> stronger fade preference,
# per "especially aggressive" reversals).
SPIKE_FADE_WEIGHT = float(os.getenv("SN89_HF_AUTO_SPIKE_FADE_WEIGHT", "0.9"))

# ── fast-path wake (user request 2026-08-31) ────────────────────────────────
# Without this, a spike-fade recognized the instant it forms still has to sit
# in the tick buffer doing nothing until the next scheduled cycle() up to
# EVAL_INTERVAL_S (5min) later — by which point the reversion this is meant
# to catch may already be over. Each tick-reader thread cheaply re-scores its
# OWN pair on every new tick (pure/read-only, same evaluate_pair() cycle()
# already uses) and wakes run_forever() immediately once a decisive spike-fade
# shows up, instead of waiting out the timer. Gating/submission itself stays
# single-threaded in cycle() — this only decides WHEN to look, never acts.
FASTPATH_ENABLED = os.getenv("SN89_HF_AUTO_FASTPATH", "1").strip().lower() not in ("0", "false", "no")
# Deliberately >= MIN_SCORE, not ==: a fast-path wake is an interrupt fired
# from a background thread on every tick, so it should only fire for a
# candidate ALREADY comfortably past the normal action bar, not merely
# scraping it — anything that clears MIN_SCORE but not this still gets acted
# on at the next regular cycle either way, this is purely a latency shortcut.
FASTPATH_MIN_SCORE = float(os.getenv("SN89_HF_AUTO_FASTPATH_MIN_SCORE", "0.35"))
# Floor between fast-path wakes for the SAME pair, so a burst spanning many
# ticks (arriving far faster than EVAL_INTERVAL_S) triggers one early cycle()
# rather than one per tick — cycle() itself re-evaluates fresh state each
# time regardless, so this only throttles how often it's asked to hurry up.
FASTPATH_COOLDOWN_S = float(os.getenv("SN89_HF_AUTO_FASTPATH_COOLDOWN_S", "5"))

# ── daily pair rotation (user request 2026-09-01) ───────────────────────────
# "Conviction over spread": trade ONE pair per UTC day instead of splitting
# attention (and the shared 30/day cap) across every board pair at once. This
# only restricts which candidate is allowed to ACT (submit) — tick-reader
# threads for every pair keep running regardless (see start()), so whichever
# pair rotates in tomorrow already has a warm buffer instead of a cold
# MIN_TICKS wait, and the dashboard can still show what every pair is doing.
ROTATE_DAILY = os.getenv("SN89_HF_AUTO_ROTATE_DAILY", "1").strip().lower() not in ("0", "false", "no")
# Rotation order, cycled by UTC day index (epoch_day % len(order)) — stateless
# on purpose: no start-date to persist or drift across restarts, at the cost
# of "day 1" landing on whatever the epoch's day-of-week happens to give it.
# Default matches the user's own walkthrough: day1 XRP, day2 BTC, day3 TAO,
# repeat. Entries not currently on the live board (or outside PAIR_ALLOWLIST)
# are skipped when computing today's pair, not treated as a wasted rotation day.
ROTATION_ORDER = [p.strip().upper() for p in
                  os.getenv("SN89_HF_AUTO_ROTATION_ORDER", "XRPUSD,BTCUSD,TAOUSD").split(",")
                  if p.strip()]


def rotation_pair_for(tradeable_pairs: list[str], t_unix: float) -> str | None:
    """The single pair allowed to trade on t_unix's UTC day: ROTATION_ORDER
    filtered down to what's actually tradeable right now, preserving
    ROTATION_ORDER's sequence. If ROTATION_ORDER shares nothing with
    `tradeable_pairs` at all (e.g. a PAIR_ALLOWLIST that doesn't overlap it),
    rotates through `tradeable_pairs` itself instead of stalling on a pool
    this bot instance can never actually trade. None only when there is
    truly nothing tradeable."""
    pool = [p for p in ROTATION_ORDER if p in tradeable_pairs] or list(tradeable_pairs)
    if not pool:
        return None
    day_index = int(t_unix // 86_400) % len(pool)
    return pool[day_index]


# ── seal-triggered fast wake (user request 2026-09-01) ──────────────────────
# "Fast decision when the prior one is sealed": once ROTATE_DAILY limits the
# bot to one pair, an open call on THAT pair is the only thing standing
# between it and its next submission for the rest of the day — worth reacting
# to within seconds rather than sitting on it for up to EVAL_INTERVAL_S. Kept
# as its own flag (not folded into FASTPATH_*) since it wakes on a call
# RESOLVING, not on a new signal forming — independent of ROTATE_DAILY too,
# since freeing up a pair sooner is strictly useful even with every pair live.
SEAL_WAKE_ENABLED = os.getenv("SN89_HF_AUTO_SEAL_WAKE", "1").strip().lower() not in ("0", "false", "no")

# ── failed-breakout reversal (EXPERIMENTAL, user request 2026-08-31) ────────
# When a call WASHES (never confirms TP, never stopped by SL) after getting
# CLOSE to TP with only a SHALLOW adverse excursion, that looks like a real
# failed breakout — price tried to confirm the move and couldn't, unlike a
# call that whipsawed through both sides before washing (which is just chop,
# not a signal). A backtest against this bot's own tiny submission history
# (7 events) was inconclusive either way — NOT proven to have positive edge.
# It ships gated the same as every other candidate (still goes through
# _gate(), still respects ROTATE_DAILY's active pair), specifically so its
# own results accumulate on the dashboard and can be judged before trusting
# it further, rather than assumed to work from a plausible-sounding story.
FAILED_BREAKOUT_ENABLED = os.getenv("SN89_HF_AUTO_FAILED_BREAKOUT", "1").strip().lower() not in ("0", "false", "no")
# "Got close to TP": the wash's own best excursion (mfe_bps) must cover at
# least this fraction of the band — the failed-breakout story only applies to
# a move that nearly confirmed, not any wash regardless of how far it got.
FAILED_BREAKOUT_MFE_RATIO = float(os.getenv("SN89_HF_AUTO_FAILED_BREAKOUT_MFE_RATIO", "0.75"))
# "Never seriously tested": the wash's worst excursion against the position
# (mae_bps) must stay under this fraction of the SL band — filters OUT the
# two-way-whipsaw washes (tested both sides hard, no directional read) that a
# bare "outcome_bps > 0" check would otherwise treat the same as a clean,
# one-directional near-miss.
FAILED_BREAKOUT_MAE_RATIO = float(os.getenv("SN89_HF_AUTO_FAILED_BREAKOUT_MAE_RATIO", "0.5"))

LF_LOCK_CACHE_TTL_S = float(os.getenv("SN89_HF_AUTO_LF_LOCK_TTL_S", "60"))
MINER_API = os.getenv("SN89_MINER_API", f"{PARTNER_API}/api/sn89/miner")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [hf-auto] {msg}", flush=True)


# ── tick buffer ──────────────────────────────────────────────────────────────
class TickBuffer:
    """Thread-safe rolling (t_ms, price) buffer for one pair."""

    def __init__(self, window_s: float):
        self._lock = threading.Lock()
        self._dq: deque[tuple[int, float]] = deque()
        self._window_ms = int(window_s * 1000)

    def append(self, t_ms: int, price: float) -> None:
        with self._lock:
            self._dq.append((int(t_ms), float(price)))
            cutoff = int(t_ms) - self._window_ms
            while self._dq and self._dq[0][0] < cutoff:
                self._dq.popleft()

    def as_ticks(self) -> list[dict]:
        with self._lock:
            return [{"t": t, "p": p} for t, p in self._dq]


def stream_reader(pair: str, token: str, buf: TickBuffer, stop_event: threading.Event,
                  on_tick=None) -> None:
    """Mirrors neurons.miner.cmd_limit's stream-watch loop: reconnect on drop,
    never raise out of the thread. `on_tick(pair, t_ms)`, if given, is called
    after every successfully buffered tick — see HfAutoBot._on_tick."""
    import requests

    url = f"{PARTNER_API}/api/sn89/closers/stream?pair={pair}"
    while not stop_event.is_set():
        try:
            with requests.get(
                url, headers={"Authorization": f"Bearer {token}"},
                stream=True, timeout=(10, 90),
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if stop_event.is_set():
                        return
                    if not line or not line.startswith(b"data:"):
                        continue
                    try:
                        d = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue
                    px = d.get("price")
                    if not px:
                        continue
                    t_ms = int(d.get("t") or d.get("ts") or time.time() * 1000)
                    buf.append(t_ms, float(px))
                    if on_tick:
                        try:
                            on_tick(pair, t_ms)
                        except Exception as e:  # noqa: BLE001 — never kill the reader over this
                            log(f"{pair}: fast-path check error: {e}")
        except Exception as e:  # noqa: BLE001 — reconnect; never kill the thread
            if not stop_event.is_set():
                log(f"{pair}: stream reconnect: {e}")
                time.sleep(3)


# ── strategy engine (tick-native, HF-horizon-sized) ─────────────────────────
def _price_at_age(ticks: list[dict], now_ms: int, age_s: float) -> float | None:
    """Last tick at or before now_ms - age_s, else the oldest tick in buffer."""
    target = now_ms - int(age_s * 1000)
    out = None
    for t in ticks:
        if t["t"] <= target:
            out = t["p"]
        else:
            break
    return out if out is not None else (ticks[0]["p"] if ticks else None)


def evaluate_pair(pair: str, ticks: list[dict], tp_bps: float, sl_bps: float,
                  horizon_s: int, now_ms: int) -> dict | None:
    """Score one pair from its own tick buffer. Returns a components dict with
    `direction` (LONG/SHORT/None) and `score` (0..1), or None if there isn't
    enough history yet to say anything.

    Deliberately NOT candle-based (LF's 15m-bar edge would throw away most of
    what a 30-120 min HF horizon can see) and NOT LLM-scored (latency/cost).
    Four ideas, same spirit as ai_finance_bot.compute_tech_edge but re-sized
    for HF's short horizon and tick granularity:
      1) fast+slow momentum, scored CONTINUOUSLY as a fraction of the pair's
         own TP band (not just sign-agreement) — a move that's already 50% of
         the way to TP means more than one that's 2% of the way, and pairs
         genuinely differ in how "loud" a real signal looks against their band,
      2) mean-reversion guard at the edge of the recent range (don't chase),
      3) aggressive-spike fade: a burst whose velocity is well above the
         established fast-window rate (not just a smooth trend's tail) is
         PREFERRED as a reversal setup, not merely discounted — see
         SPIKE_ACCEL_RATIO's docstring for how this is told apart from (1),
      4) volatility-vs-band gate: too quiet -> likely wash; too extended ->
         the easy room to TP is already spent.
    """
    ticks = sorted(ticks, key=lambda t: t["t"])
    if len(ticks) < MIN_TICKS:
        return None

    # Momentum windows scale with THIS pair's own horizon (see REF_HORIZON_S):
    # a 7200s pair reads momentum over windows 4x wider than a 1800s one, so
    # the same MOM_FAST_S/MOM_SLOW_S knobs stay meaningful across the board
    # instead of only fitting whichever horizon they were tuned against.
    scale = (horizon_s or REF_HORIZON_S) / REF_HORIZON_S
    mom_fast_s = MOM_FAST_S * scale
    mom_slow_s = MOM_SLOW_S * scale

    price_now = ticks[-1]["p"]
    p_fast = _price_at_age(ticks, now_ms, mom_fast_s)
    p_slow = _price_at_age(ticks, now_ms, mom_slow_s)
    mom_fast_bps = (price_now - p_fast) / p_fast * 10_000 if p_fast else 0.0
    mom_slow_bps = (price_now - p_slow) / p_slow * 10_000 if p_slow else 0.0

    prices = [t["p"] for t in ticks]
    hi, lo = max(prices), min(prices)
    mid = (hi + lo) / 2 or price_now
    realized_range_bps = (hi - lo) / mid * 10_000 if mid else 0.0
    vol_ratio = (realized_range_bps / tp_bps) if tp_bps else 0.0
    swing_pos = (price_now - lo) / (hi - lo) if hi > lo else 0.5
    extension_ratio = (abs(mom_slow_bps) / tp_bps) if tp_bps else 0.0

    comps = {
        "mom_fast_bps": round(mom_fast_bps, 2), "mom_slow_bps": round(mom_slow_bps, 2),
        "realized_range_bps": round(realized_range_bps, 2),
        "vol_ratio": round(vol_ratio, 3), "swing_pos": round(swing_pos, 3),
        "extension_ratio": round(extension_ratio, 3),
    }

    # norm_* is each window's move as a fraction of the pair's OWN tp_bps band,
    # clamped to +-1. long_s/short_s blend fast+slow evenly (0..1 each) so two
    # pairs with the same sign-agreement but different move sizes no longer tie
    # at the same score — which is what let one pair (whichever iterates first)
    # win "best candidate" by iteration order alone under the old step scheme.
    norm_fast = max(-1.0, min(1.0, mom_fast_bps / tp_bps)) if tp_bps else 0.0
    norm_slow = max(-1.0, min(1.0, mom_slow_bps / tp_bps)) if tp_bps else 0.0
    long_s = 0.5 * max(0.0, norm_fast) + 0.5 * max(0.0, norm_slow)
    short_s = 0.5 * max(0.0, -norm_fast) + 0.5 * max(0.0, -norm_slow)

    # Aggressive-spike fade. recent_v vs base_v is what tells a genuine burst
    # apart from the tail of an already-smooth trend: a constant-rate move
    # reads recent_v ~= base_v (ratio ~= 1, e.g. test_sustained_uptrend), while
    # a burst concentrated in the last SPIKE_WINDOW_S — even one fully inside
    # the same mom_fast window, so momentum "sees" it too — reads recent_v far
    # above the window's own average rate. On a hit, the chase direction is
    # DAMPENED (not just left alone: mom_fast/mom_slow unavoidably contain the
    # same burst and would otherwise still win) and the fade direction is
    # floored at a score scaled by how big the burst is, so it wins over the
    # dampened chase — "prefer reversal, especially aggressive reversal", not
    # merely "don't chase as hard".
    p_spike = _price_at_age(ticks, now_ms, SPIKE_WINDOW_S)
    spike_bps = (price_now - p_spike) / p_spike * 10_000 if p_spike else 0.0
    spike_ratio = (abs(spike_bps) / tp_bps) if tp_bps else 0.0
    base_v = (abs(mom_fast_bps) / mom_fast_s) if mom_fast_s else 0.0
    recent_v = abs(spike_bps) / SPIKE_WINDOW_S
    accel_ratio = (recent_v / base_v) if base_v > 1e-9 else (float("inf") if recent_v > 0 else 0.0)
    comps["spike_bps"] = round(spike_bps, 2)
    comps["spike_ratio"] = round(spike_ratio, 3)
    comps["spike_accel"] = round(accel_ratio, 2) if accel_ratio != float("inf") else None

    if spike_ratio >= SPIKE_MIN_RATIO and accel_ratio >= SPIKE_ACCEL_RATIO:
        fade = min(1.0, spike_ratio) * SPIKE_FADE_WEIGHT
        if spike_bps > 0:
            long_s *= SPIKE_CHASE_DAMPEN
            short_s = max(short_s, fade)
            comps["spike_fade"] = "short_after_up_spike"
        else:
            short_s *= SPIKE_CHASE_DAMPEN
            long_s = max(long_s, fade)
            comps["spike_fade"] = "long_after_down_spike"

    # Mean-reversion guard: fade conviction if price already sits at the
    # extreme of its own recent range in the direction we'd be calling —
    # that is exactly the "chasing an extended move" trap a fixed 1R band
    # punishes hardest.
    if long_s > short_s and swing_pos > 0.85:
        long_s *= 0.5
        comps["guard"] = "long_at_range_high"
    if short_s > long_s and swing_pos < 0.15:
        short_s *= 0.5
        comps["guard"] = "short_at_range_low"

    if long_s - short_s >= MIN_EDGE:
        direction, raw = "LONG", long_s
    elif short_s - long_s >= MIN_EDGE:
        direction, raw = "SHORT", short_s
    else:
        return {"pair": pair, "direction": None, "score": 0.0, "price": price_now,
                "components": comps, "reason": "no_edge"}

    if vol_ratio < MIN_VOL_RATIO:
        return {"pair": pair, "direction": None, "score": 0.0, "price": price_now,
                "components": comps, "reason": f"too_quiet:{vol_ratio:.3f}<{MIN_VOL_RATIO}"}
    if extension_ratio > MAX_EXTENSION_RATIO:
        raw *= 0.4
        comps["extended"] = True

    score = max(0.0, min(1.0, raw))
    return {"pair": pair, "direction": direction, "score": round(score, 3),
            "price": price_now, "components": comps, "reason": "ok"}


def should_fastpath_wake(ticks: list[dict], tp_bps: float, sl_bps: float,
                         horizon_s: int, now_ms: int,
                         min_score: float = FASTPATH_MIN_SCORE) -> bool:
    """True if scoring THIS tick buffer right now already shows a decisive
    aggressive-spike-fade — the tick-reader thread's cue to wake run_forever()
    early rather than let a live reversal sit unacted-on until the next
    scheduled cycle(). Pure/read-only: reuses evaluate_pair verbatim, never
    gates or submits itself."""
    res = evaluate_pair("_fastpath", ticks, tp_bps, sl_bps, horizon_s, now_ms)
    if not res or not res.get("direction"):
        return False
    return bool(res["components"].get("spike_fade")) and res["score"] >= min_score


def call_just_sealed(direction: str, entry, tp_bps: float, sl_bps: float, t0_ms: int,
                     horizon_s: int, ticks_sorted: list[dict], now_ms: int) -> bool:
    """True if this call's open-until point — the SAME `hf.open_until_ms` the
    consensus pair-open gate replays — has already passed as of now_ms. Pure:
    the caller (HfAutoBot._check_seal_wake) owns per-call dedup so this only
    triggers a wake once per call, not on every tick after it seals."""
    return hf.open_until_ms(direction, entry, tp_bps, sl_bps, t0_ms, horizon_s, ticks_sorted) <= now_ms


def failed_breakout_reversal(direction: str, entry, tp_bps: float, sl_bps: float,
                             t0_ms: int, horizon_s: int, ticks_sorted: list[dict],
                             mfe_ratio: float = FAILED_BREAKOUT_MFE_RATIO,
                             mae_ratio: float = FAILED_BREAKOUT_MAE_RATIO) -> str | None:
    """EXPERIMENTAL (see FAILED_BREAKOUT_* docstring above). Only called on a
    call that has ALREADY sealed — this re-derives whether it sealed via a
    WASH (using hf.grade, the same authoritative status logic the pair-open
    gate mirrors elsewhere) rather than a decisive touch, then checks the
    shape of that wash: got close to TP (>= mfe_ratio of the band) but never
    seriously threatened the SL (<= mae_ratio of the band). Returns the
    reversal direction if that shape matches, else None. Pure/read-only."""
    if entry is None or float(entry) <= 0 or tp_bps <= 0 or sl_bps <= 0:
        return None
    grading = hf.grade("_failed_breakout_check", direction, float(entry), tp_bps, sl_bps,
                       t0_ms, horizon_s, ticks_sorted)
    if grading.get("status") != "wash":
        return None
    up = 1 if direction == "LONG" else -1
    t0_ms, t_end = int(t0_ms), int(t0_ms) + int(horizon_s) * 1000
    moves = [
        (float(t["p"]) - float(entry)) / float(entry) * 10_000 * up
        for t in ticks_sorted if t0_ms < int(t["t"]) <= t_end
    ]
    if not moves:
        return None
    mfe_bps, mae_bps = max(moves), min(moves)
    if mfe_bps >= mfe_ratio * tp_bps and abs(mae_bps) <= mae_ratio * sl_bps:
        return "SHORT" if direction == "LONG" else "LONG"
    return None


# ── local safety gate (pure functions — unit-tested without network) ───────
def parse_manual_request(req: dict, board: dict) -> tuple[str, str] | None:
    """Validate a dashboard manual-submit request against the live HF board.
    Pure so it's testable without a running bot/wallet — the only reason a
    manual submit should ever get silently dropped instead of attempted."""
    pair = str(req.get("pair", "")).upper().strip()
    direction = str(req.get("direction", "")).upper().strip()
    if direction not in ("LONG", "SHORT"):
        return None
    if pair not in board:
        return None
    return pair, direction


def local_rate_gate(submit_log_ms: list[int], now_ms: int, t0_unix: float) -> str | None:
    """None if a submit right now would pass the consensus rate rule
    (`hf.check_rate`); else the refusal reason."""
    try:
        hf.check_rate(submit_log_ms, now_ms, t0_unix)
    except hf.HFRejected as e:
        return str(e)
    return None


def local_pace_gate(submit_log_ms: list[int], now_ms: int, daily_cap: int,
                     slack: float = PACE_SLACK) -> str | None:
    """None if a submit right now stays within the pro-rated share of
    `daily_cap` for how far the UTC day has progressed; else a reason
    naming the ETA of the next allowed slot.

    This is a LOCAL pacing choice, not a consensus rule (`hf.check_rate`'s
    30/day + 250ms-gap are the only server-enforced limits) — its only job
    is to stop this bot's own strategy from racing the shared daily cap to
    zero in the first hour of a choppy day and then sitting silent for the
    other 23, which starves diversity (fewer distinct-day/asset outcomes)
    for no protocol benefit.
    """
    if daily_cap <= 0:
        return None
    day_ms = 86_400_000
    day_start = (now_ms // day_ms) * day_ms
    day_frac = (now_ms - day_start) / day_ms
    submits_today = sum(1 for t in submit_log_ms if t >= day_start)
    allowed_by_now = day_frac * daily_cap + slack
    if submits_today < allowed_by_now:
        return None
    # Absolute clock time, NOT "minutes from now": this must read IDENTICAL
    # across repeated cycles while submits_today hasn't changed, or an
    # ever-ticking countdown would defeat cycle()'s decision-dedup the same
    # way it does for every other gate reason (see _last_decision_key).
    slot_frac = max(0.0, (submits_today - slack) / daily_cap)
    eta_ms = day_start + slot_frac * day_ms
    eta_str = time.strftime("%H:%M UTC", time.gmtime(eta_ms / 1000))
    return f"pace:{submits_today}/{daily_cap} used today, next slot ~{eta_str}"


def local_pair_open_gate(open_calls: list[dict], now_ms: int, t0_unix: float) -> str | None:
    """`open_calls`: this pair's own accepted, non-void HF calls, each
    {direction, entry, tp_bps, sl_bps, t0_ms, horizon_s, ticks}. `ticks` is
    whatever of our own tick buffer we captured since that call's t0 — a
    truncated series simply reads as "still open", the same conservative
    reading `hf.open_until_ms` documents for a live caller.
    """
    prior_open_until = [
        hf.open_until_ms(c["direction"], c["entry"], c["tp_bps"], c["sl_bps"],
                         c["t0_ms"], c["horizon_s"], c["ticks"])
        for c in open_calls
    ]
    try:
        hf.check_pair_open(prior_open_until, now_ms, t0_unix)
    except hf.HFRejected as e:
        return str(e)
    return None


def confirm_streak(prev: dict | None, direction: str | None) -> tuple[dict | None, int]:
    """Pure streak counter: how many consecutive calls (including this one) have
    reported the SAME direction for one pair. `prev` is whatever this function
    returned last time for that pair (None initially); pass the returned state
    back in next cycle. A None direction (no_edge/too_quiet/etc.) resets to 0 —
    a candidate must be actionable on its own each cycle to keep its streak alive.
    """
    if direction is None:
        return None, 0
    n = prev["n"] + 1 if prev and prev.get("direction") == direction else 1
    return {"direction": direction, "n": n}, n


def local_cross_lock_gate(pair: str, lf_last_t0_ms: dict, now_ms: int) -> str | None:
    """Mirrors `hf.is_pair_locked` from the HF side: block if OUR OWN LF call
    on this pair landed inside the rolling 24h window. `lf_last_t0_ms`:
    {pair: last LF t0_unix_ms}, sourced from the public miner API (the local
    LF submit log has no per-pair field to check against)."""
    last = lf_last_t0_ms.get(pair)
    if last is None:
        return None
    if 0 <= now_ms - int(last) < hf.PAIR_LOCK_MS:
        return f"cross_mechanism_lock:{pair}:{int(last) + hf.PAIR_LOCK_MS}"
    return None


# ── persistence helpers ──────────────────────────────────────────────────────
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def load_own_receipts(hk: str) -> list[dict]:
    """Own accepted HF receipts, oldest first — the audit trail submit_hf()
    already appends to on every call."""
    path = _receipts_path(hk)
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return out


def enabled() -> bool:
    """No flag file yet = enabled by default (first run). Delete/recreate
    yourself, or use the dashboard toggle."""
    if not ENABLE_FLAG_PATH.exists():
        return True
    return load_json(ENABLE_FLAG_PATH, {"enabled": True}).get("enabled", True)


def set_enabled(value: bool) -> None:
    save_json(ENABLE_FLAG_PATH, {"enabled": bool(value), "at": time.time()})


# ── the bot ──────────────────────────────────────────────────────────────────
class HfAutoBot:
    def __init__(self, wallet, dry_run: bool):
        self.wallet = wallet
        self.hotkey = wallet.hotkey.ss58_address
        self.dry_run = dry_run
        self.buffers: dict[str, TickBuffer] = {}
        self.stop_event = threading.Event()
        self.submit_log: list[int] = load_json(_submit_log_path(self.hotkey), [])
        # Entry price for OUR OWN calls, keyed by receipt seq, persisted across
        # restarts. hf.price_at() needs a tick at/before t0_ms in the CURRENT
        # in-memory buffer to compute it — a restart wipes that buffer, so
        # without this cache every call opened before a restart would read
        # back as entry=None. hf.open_until_ms() treats "no entry price" as
        # VOID (holds nothing), which would make local_pair_open_gate
        # silently free up a pair that's still genuinely open on-chain right
        # after any restart. This cache is what keeps that gate correct.
        self._entry_cache: dict[int, float] = {
            int(k): v for k, v in load_json(_entry_cache_path(self.hotkey), {}).items()
        }
        self.decisions: deque[dict] = deque(maxlen=100)
        self._last_decision_key: tuple | None = None
        self._lf_lock_cache: dict[str, int] = {}
        self._lf_lock_cache_at = 0.0
        self._streaks: dict[str, dict] = {}
        # See _on_tick / run_forever — lets a fast-path spike-fade interrupt
        # the EVAL_INTERVAL_S sleep instead of waiting it out.
        self._wake_event = threading.Event()
        self._last_fastpath_wake: dict[str, float] = {}
        # See _check_seal_wake — cycle()'s own copy of open_by_pair, read
        # (never written) by the tick-reader threads. t0_ms of the last call
        # we've already woken the loop for, per pair, so a sealed call that
        # cycle() hasn't cleared out yet doesn't re-wake on every tick.
        self._last_open_by_pair: dict[str, list[dict]] = {}
        self._sealed_announced: dict[str, int] = {}
        self._last_manual_check = 0.0
        # Set by _check_seal_wake (tick-reader thread) when a just-sealed call
        # matches the failed-breakout shape; consumed once by cycle() (main
        # thread) via _consume_forced_candidate. Plain unlocked attribute —
        # same reasoning as _last_open_by_pair above: a single dict reference
        # assignment is atomic enough under the GIL, and a lost race just
        # means the signal is picked up next cycle instead of this one.
        self._forced_candidate: dict | None = None

    # ── setup ────────────────────────────────────────────────────────────
    def _pairs(self) -> list[str]:
        board = hf.hf_bands_as_of(time.time()) or {}
        pairs = sorted(board)
        if PAIR_ALLOWLIST:
            pairs = [p for p in pairs if p in PAIR_ALLOWLIST]
        return pairs

    def start(self) -> None:
        from neurons.miner import mint_miner_token

        token = mint_miner_token(self.wallet)
        pairs = self._pairs()
        if not pairs:
            raise RuntimeError("no HF pairs to trade (board empty or allowlist matched nothing)")
        log(f"hotkey={self.hotkey[:8]}… pairs={pairs} dry_run={self.dry_run}")
        for pair in pairs:
            buf = TickBuffer(BUFFER_WINDOW_S)
            self.buffers[pair] = buf
            on_tick = self._on_tick if (FASTPATH_ENABLED or SEAL_WAKE_ENABLED) else None
            th = threading.Thread(
                target=stream_reader, args=(pair, token, buf, self.stop_event, on_tick),
                daemon=True, name=f"hf-tick-{pair}",
            )
            th.start()

    def _on_tick(self, pair: str, t_ms: int) -> None:
        """Runs on a tick-reader thread, NOT the main cycle() thread — both
        checks below must stay read-only w.r.t. anything cycle() itself
        writes (see should_fastpath_wake / _check_seal_wake)."""
        if SEAL_WAKE_ENABLED:
            self._check_seal_wake(pair, t_ms)
        now = time.time()
        # Cheap stat() throttled to ~1/s (ticks can arrive many times a
        # second per pair) — lets a dashboard manual submit resolve within a
        # second or two instead of waiting out the rest of EVAL_INTERVAL_S.
        if now - self._last_manual_check >= 1.0:
            self._last_manual_check = now
            if MANUAL_REQUEST_PATH.exists():
                self._wake_event.set()
        if not FASTPATH_ENABLED:
            return
        if now - self._last_fastpath_wake.get(pair, 0.0) < FASTPATH_COOLDOWN_S:
            return
        board = hf.hf_bands_as_of(now) or {}
        row = board.get(pair)
        buf = self.buffers.get(pair)
        if not row or not buf:
            return
        tp_bps, sl_bps, horizon_s, _cls = row
        if should_fastpath_wake(buf.as_ticks(), tp_bps, sl_bps, horizon_s, t_ms):
            self._last_fastpath_wake[pair] = now
            log(f"{pair}: fast-path wake — aggressive spike-fade, not waiting for the next cycle")
            self._wake_event.set()

    def _check_seal_wake(self, pair: str, t_ms: int) -> None:
        """Wakes run_forever() the instant a call cycle() already knows about
        seals (decisive touch or horizon wash) — with ROTATE_DAILY on, an open
        call on today's pair is the only thing blocking the next submission
        for the rest of the day, so this is worth checking on every tick."""
        calls = self._last_open_by_pair.get(pair)
        if not calls:
            return
        buf = self.buffers.get(pair)
        if not buf:
            return
        ticks = sorted(buf.as_ticks(), key=lambda t: t["t"])
        for call in calls:
            t0_ms = call["t0_ms"]
            if self._sealed_announced.get(pair) == t0_ms:
                continue
            if call_just_sealed(call["direction"], call.get("entry"), call["tp_bps"],
                               call["sl_bps"], t0_ms, call["horizon_s"], ticks, t_ms):
                self._sealed_announced[pair] = t0_ms
                log(f"{pair}: prior call sealed — waking immediately for the next decision")
                if FAILED_BREAKOUT_ENABLED:
                    reversal = failed_breakout_reversal(
                        call["direction"], call.get("entry"), call["tp_bps"],
                        call["sl_bps"], t0_ms, call["horizon_s"], ticks)
                    if reversal:
                        log(f"{pair}: failed-breakout wash detected — queuing "
                            f"{reversal} reversal candidate")
                        self._forced_candidate = {"pair": pair, "direction": reversal}
                self._wake_event.set()

    def stop(self) -> None:
        self.stop_event.set()
        self._wake_event.set()

    # ── open-position bookkeeping (own receipts, replayed through hf.py) ──
    def _open_calls_by_pair(self, now_ms: int) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        cache_dirty = False
        for row in load_own_receipts(self.hotkey):
            sub = row.get("submit") or {}
            rcpt = row.get("receipt") or {}
            payload = sub.get("payload") or {}
            if str(payload.get("kind", "")) == "closers":
                continue
            pair = str(payload.get("trade_pair") or "").upper()
            t0_ms = rcpt.get("grid_t0_ms")
            if not pair or not t0_ms:
                continue
            horizon_s = int(payload.get("horizon_s") or 0)
            if not horizon_s or now_ms >= int(t0_ms) + horizon_s * 1000 + 5000:
                continue  # long past its own horizon — cannot still be open
            ticks = self.buffers.get(pair).as_ticks() if pair in self.buffers else []
            entry = hf.price_at(sorted(ticks, key=lambda t: t["t"]), int(t0_ms))
            seq = sub.get("seq")
            if entry is not None and seq is not None:
                if self._entry_cache.get(int(seq)) != entry:
                    cache_dirty = True
                self._entry_cache[int(seq)] = entry
            elif entry is None and seq is not None:
                entry = self._entry_cache.get(int(seq))
            out.setdefault(pair, []).append({
                "direction": str(payload.get("direction") or "LONG").upper(),
                "entry": entry,
                "tp_bps": float(payload.get("tp_bps") or 0),
                "sl_bps": float(payload.get("sl_bps") or 0),
                "t0_ms": int(t0_ms),
                "horizon_s": horizon_s,
                "ticks": ticks,
            })
        if cache_dirty:
            # Never pruned, but grows at most <=30/day (the submission cap),
            # so even years of history stays a trivially small file.
            save_json(_entry_cache_path(self.hotkey), self._entry_cache)
        return out

    def _consume_manual_request(self, board: dict) -> dict | None:
        """Pop the dashboard's manual-submit request for THIS instance, if
        any. One-shot (read-then-delete) so a restart or two overlapping
        cycle() calls can never replay the same click twice. Returns a
        best-shaped dict (pair/direction/score/price) ready to feed straight
        into the normal gate+submit path below, or None if there was nothing
        queued or it didn't validate."""
        if not MANUAL_REQUEST_PATH.exists():
            return None
        req = load_json(MANUAL_REQUEST_PATH, {})
        try:
            MANUAL_REQUEST_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        parsed = parse_manual_request(req, board)
        if parsed is None:
            pair = str(req.get("pair", "")).upper().strip()
            direction = str(req.get("direction", "")).upper().strip()
            self._record_decision(pair=pair, direction=direction, score=1.0, entry_ref=None,
                                  tp_bps=0, sl_bps=0, horizon_s=0, status="error", manual=True,
                                  source="manual",
                                  reason=f"manual_submit: invalid or unknown pair {pair!r}")
            return None
        pair, direction = parsed
        price = None
        buf = self.buffers.get(pair)
        if buf:
            ticks = sorted(buf.as_ticks(), key=lambda t: t["t"])
            if ticks:
                price = ticks[-1]["p"]
        return {"pair": pair, "direction": direction, "score": 1.0, "price": price}

    def _consume_forced_candidate(self, board: dict, active_pair: str | None) -> dict | None:
        """Pop the failed-breakout reversal queued by _check_seal_wake, if
        any. One-shot, and re-validated here (not just when it was queued) —
        the board or today's rotation slot could have moved on by the time
        cycle() gets around to it. Unlike a dashboard manual request, this
        STILL respects ROTATE_DAILY: it's an automated signal, not a human
        override, so it can only act on today's pair like everything else."""
        forced = self._forced_candidate
        self._forced_candidate = None
        if not forced:
            return None
        pair, direction = forced["pair"], forced["direction"]
        if pair not in board:
            return None
        if active_pair is not None and pair != active_pair:
            return None
        price = None
        buf = self.buffers.get(pair)
        if buf:
            ticks = sorted(buf.as_ticks(), key=lambda t: t["t"])
            if ticks:
                price = ticks[-1]["p"]
        return {"pair": pair, "direction": direction, "score": 1.0, "price": price}

    def _refresh_lf_lock_cache(self) -> None:
        import urllib.error
        import urllib.request

        now = time.time()
        if now - self._lf_lock_cache_at < LF_LOCK_CACHE_TTL_S:
            return
        self._lf_lock_cache_at = now
        try:
            req = urllib.request.Request(
                f"{MINER_API.rstrip('/')}/{self.hotkey}",
                headers={"User-Agent": "sn89-hf-auto-bot"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            calls = data.get("calls") or []
            latest: dict[str, int] = {}
            for c in calls:
                pair = str(c.get("trade_pair") or "").upper()
                t0 = c.get("t0_unix")
                if not pair or not t0:
                    continue
                t0_ms = int(float(t0) * 1000)
                if t0_ms > latest.get(pair, -1):
                    latest[pair] = t0_ms
            self._lf_lock_cache = latest
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as e:
            log(f"LF lock cache refresh failed (keeping stale cache): {e}")

    # ── gate + submit ────────────────────────────────────────────────────
    def _gate(self, pair: str, now_ms: int, open_by_pair: dict[str, list[dict]],
              skip_pace: bool = False) -> str | None:
        t0_unix = now_ms / 1000.0
        r = local_rate_gate(self.submit_log, now_ms, t0_unix)
        if r:
            return r
        # skip_pace=True is ONLY set for a dashboard manual submit: pacing is
        # a self-imposed throttle to spread OUR OWN candidates across the
        # day, not a consensus rule — a human explicitly clicking "submit
        # now" has already made that timing call. rate/pair-open/cross-lock
        # below are real consensus rules the ingest would refuse on, so
        # those still apply even to a manual submit.
        if PACE_ENABLED and not skip_pace:
            daily_cap = hf.hf_rules_as_of(t0_unix)[0]
            r = local_pace_gate(self.submit_log, now_ms, daily_cap)
            if r:
                return r
        r = local_pair_open_gate(open_by_pair.get(pair, []), now_ms, t0_unix)
        if r:
            return r
        return local_cross_lock_gate(pair, self._lf_lock_cache, now_ms)

    def _record_decision(self, **kv) -> None:
        kv["at"] = time.time()
        self.decisions.appendleft(kv)

    def _write_state(self, candidates: list[dict],
                      open_by_pair: dict[str, list[dict]] | None = None,
                      active_pair: str | None = None) -> None:
        # open_calls is read straight from our own receipts file every cycle
        # (via _open_calls_by_pair), NOT reconstructed from the `decisions`
        # log — that deque only holds the last 100 entries and a single
        # locked pair re-logging "blocked" every ~1.5s cycle can fully evict
        # its own original "submitted" entry (and everything else) within a
        # couple of minutes. This is the durable source of truth the
        # dashboard's open-positions panel reads from.
        #
        # _open_calls_by_pair only drops calls whose full horizon has
        # elapsed — it does NOT know about early decisive touches (that's
        # local_pair_open_gate's job, via hf.open_until_ms). Re-apply that
        # same touch-aware check here too, or a call that TP'd/SL'd in the
        # first few minutes of a 30-min horizon would keep showing as "open"
        # in the dashboard for the rest of that horizon even though the gate
        # itself already considers the pair free again.
        now_ms = int(time.time() * 1000)
        open_calls = [
            {"pair": pair, "direction": c["direction"], "entry": c["entry"],
             "tp_bps": c["tp_bps"], "sl_bps": c["sl_bps"],
             "t0_ms": c["t0_ms"], "horizon_s": c["horizon_s"]}
            for pair, calls in (open_by_pair or {}).items() for c in calls
            if hf.open_until_ms(c["direction"], c["entry"], c["tp_bps"], c["sl_bps"],
                                c["t0_ms"], c["horizon_s"], c["ticks"]) > now_ms
        ]
        save_json(STATE_PATH, {
            "updated_at": time.time(),
            "hotkey": self.hotkey,
            "enabled": enabled(),
            "dry_run": self.dry_run,
            "pairs": sorted(self.buffers),
            "rotation": {"enabled": ROTATE_DAILY, "active_pair": active_pair,
                         "order": ROTATION_ORDER} if ROTATE_DAILY else {"enabled": False},
            "candidates": candidates,
            "open_calls": open_calls,
            "decisions": list(self.decisions),
            "submits_today": sum(
                1 for t in self.submit_log
                if int(t) // 86_400_000 == int(time.time() * 1000) // 86_400_000
            ),
        })

    def cycle(self) -> None:
        now_ms = int(time.time() * 1000)
        board = hf.hf_bands_as_of(now_ms / 1000.0) or {}
        # Checked BEFORE the enabled() early-return and BEFORE ROTATE_DAILY /
        # MIN_SCORE / CONFIRM_CYCLES below: a manual submit is an explicit
        # human decision for a specific pair+direction right now, not another
        # vote in the strategy's own candidate scoring — pausing the auto
        # loop or today's rotation slot shouldn't block a deliberate click.
        manual = self._consume_manual_request(board)

        if not enabled() and not manual:
            self._write_state(candidates=[])
            return

        self._refresh_lf_lock_cache()
        open_by_pair = self._open_calls_by_pair(now_ms)
        # Read by _check_seal_wake on the tick-reader threads — see its
        # docstring for why this is a plain unlocked assignment.
        self._last_open_by_pair = open_by_pair

        active_pair = rotation_pair_for(list(self.buffers), now_ms / 1000.0) if ROTATE_DAILY else None
        # Unlike manual, a forced candidate (failed-breakout reversal) is an
        # automated signal, not a human override — it still respects today's
        # rotation slot, so only consumed once active_pair is known, and
        # skipped entirely if a manual request already won this cycle.
        forced = None if manual else self._consume_forced_candidate(board, active_pair)

        candidates = []
        for pair, buf in self.buffers.items():
            row = board.get(pair)
            if not row:
                continue
            tp, sl, hz, _cls = row
            res = evaluate_pair(pair, buf.as_ticks(), tp, sl, hz, now_ms)
            if res:
                self._streaks[pair], res["streak"] = confirm_streak(
                    self._streaks.get(pair), res["direction"])
                candidates.append(res)
        self._write_state(candidates=candidates, open_by_pair=open_by_pair, active_pair=active_pair)

        if manual:
            best, source = manual, "manual"
        elif forced:
            best, source = forced, "failed_breakout"
        else:
            # CONFIRM_CYCLES gates ACTION (gating + submission), not visibility —
            # every raw candidate above is already in the state file regardless
            # of streak. ROTATE_DAILY gates it the same way: other pairs still
            # SCORE (so the dashboard shows what they're doing) but can never be
            # the one that acts.
            scored = [c for c in candidates if c["direction"] and c["score"] >= MIN_SCORE
                      and c.get("streak", 0) >= CONFIRM_CYCLES
                      and (active_pair is None or c["pair"] == active_pair)]
            if not scored:
                return
            best, source = max(scored, key=lambda c: c["score"]), "auto"
        pair, direction = best["pair"], best["direction"]
        tp, sl, hz, _cls = board[pair]
        common = dict(pair=pair, direction=direction, score=best["score"],
                      entry_ref=best.get("price"), tp_bps=tp, sl_bps=sl, horizon_s=hz,
                      manual=bool(manual), source=source)

        # Only a manual dashboard click skips pacing (see _gate's docstring) —
        # a failed-breakout reversal is an automated candidate like any other
        # and stays subject to it.
        reason = self._gate(pair, now_ms, open_by_pair, skip_pace=bool(manual))
        if reason:
            # Dedup: a locked pair re-gates with the SAME reason every cycle
            # for its whole horizon — logging that every ~1.5s would flood
            # the 100-entry decisions deque and evict genuinely new events
            # (submitted/refused/error, or a different pair's history)
            # within a couple of minutes. Only log on a state CHANGE. A
            # manual click always logs though — the human is waiting on it.
            key = (pair, "blocked", reason)
            if manual or key != self._last_decision_key:
                self._record_decision(**common, status="blocked", reason=reason)
            self._last_decision_key = key
            return
        self._last_decision_key = None

        tag = f" ({source})" if source != "auto" else ""
        if self.dry_run:
            self._record_decision(**common, status="dry_run", reason="")
            log(f"DRY-RUN would submit {pair} {direction} score={best['score']}{tag}")
            return

        from neurons.miner import submit_hf

        try:
            resp = submit_hf(self.wallet, pair, direction)
        except Exception as e:  # noqa: BLE001
            self._record_decision(**common, status="error", reason=str(e))
            log(f"{pair} {direction}: submit_hf error: {e}")
            return

        if resp.get("kind") == "hf.receipt":
            self.submit_log.append(now_ms)
            save_json(_submit_log_path(self.hotkey), self.submit_log[-2000:])
            self._record_decision(**common, status="submitted", reason="")
            log(f"→ SUBMITTED {pair} {direction} score={best['score']}{tag}")
        else:
            self._record_decision(**common, status="refused",
                                  reason=resp.get("reason", "unknown"))
            log(f"{pair} {direction}: refused: {resp.get('reason', 'unknown')}")

    def run_forever(self) -> None:
        self.start()
        try:
            while not self.stop_event.is_set():
                try:
                    self.cycle()
                except Exception as e:  # noqa: BLE001 — never let one bad cycle kill the daemon
                    log(f"cycle error: {type(e).__name__}: {e}")
                # Sleep EVAL_INTERVAL_S as normal, UNLESS a tick-reader thread
                # sets _wake_event first (see _on_tick) — then loop back into
                # cycle() immediately instead of sitting on a stale spike-fade
                # for however much of the interval is left. cycle() itself is
                # unchanged either way: this only decides when it gets called.
                self._wake_event.wait(timeout=EVAL_INTERVAL_S)
                self._wake_event.clear()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main() -> int:
    p = argparse.ArgumentParser(description="SN89 HF auto-bot (tick-native, no LLM)")
    p.add_argument("--wallet.name", dest="wallet_name", default=WALLET_NAME)
    p.add_argument("--wallet.hotkey", dest="wallet_hotkey", default=WALLET_HOTKEY)
    p.add_argument("--live", action="store_true", help="actually call submit_hf")
    p.add_argument("--dry-run", action="store_true", help="log decisions, never submit")
    p.add_argument("--once", action="store_true", help="one cycle then exit (after warmup)")
    p.add_argument("--warmup-s", type=float, default=20.0,
                   help="seconds to let tick buffers fill before the first --once cycle")
    args = p.parse_args()

    dry = args.dry_run or not args.live
    if dry:
        log("DRY-RUN mode (pass --live to submit for real)")

    import bittensor as bt

    wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    bot = HfAutoBot(wallet, dry_run=dry)

    if args.once:
        bot.start()
        time.sleep(args.warmup_s)
        bot.cycle()
        bot.stop()
        return 0

    bot.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
