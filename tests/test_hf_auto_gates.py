"""bots/hf_auto_bot.py — local safety-gate + strategy-engine unit tests.

Pure-function tests only, no network and no wallet: these are the functions
that must agree with `sn89_signals/hf.py`'s consensus rules closely enough
that the auto-bot never wastes a submission or gets silently refused.
"""
from bots.hf_auto_bot import (
    call_just_sealed,
    confirm_streak,
    evaluate_pair,
    failed_breakout_reversal,
    local_cross_lock_gate,
    local_pace_gate,
    local_pair_open_gate,
    local_rate_gate,
    parse_manual_request,
    rotation_pair_for,
    should_fastpath_wake,
)
from sn89_signals import hf

DAY_MS = 86_400_000


class TestLocalRateGate:
    def test_allows_first_of_the_day(self):
        assert local_rate_gate([], 10 * DAY_MS, 10 * DAY_MS / 1000) is None

    def test_thirtieth_ok_thirtyfirst_blocked(self):
        base = 10 * DAY_MS
        prior = [base + i * 60_000 for i in range(29)]
        t0_unix = (base + 29 * 60_000) / 1000
        assert local_rate_gate(prior, base + 29 * 60_000, t0_unix) is None

        prior30 = [base + i * 60_000 for i in range(30)]
        t_next = base + 30 * 60_000
        reason = local_rate_gate(prior30, t_next, t_next / 1000)
        assert reason is not None and "daily_cap" in reason

    def test_min_gap_250ms(self):
        prior = [10 * DAY_MS]
        t0_unix = (10 * DAY_MS + 249) / 1000
        reason = local_rate_gate(prior, 10 * DAY_MS + 249, t0_unix)
        assert reason is not None and "min_gap" in reason
        assert local_rate_gate(prior, 10 * DAY_MS + 250, (10 * DAY_MS + 250) / 1000) is None

    def test_resets_on_utc_day_boundary(self):
        prior = [10 * DAY_MS + i * 60_000 for i in range(30)]
        assert local_rate_gate(prior, 11 * DAY_MS, 11 * DAY_MS / 1000) is None


class TestLocalPaceGate:
    """Spreads the shared 30/day cap across the UTC day so a choppy first
    hour can't exhaust it and leave the bot dark for the rest of the day."""

    CAP = 30

    def test_empty_day_is_never_paced(self):
        # day_frac=0 -> allowed_by_now == slack (>0), 0 submits is always under it
        assert local_pace_gate([], 10 * DAY_MS, self.CAP) is None

    def test_on_schedule_mid_day_is_not_paced(self):
        day_start = 10 * DAY_MS
        now = day_start + DAY_MS // 2  # halfway through the day
        # 14 submits by noon vs a 15-by-noon pro-rated budget (+slack) -> ok
        prior = [day_start + i * 1000 for i in range(14)]
        assert local_pace_gate(prior, now, self.CAP) is None

    def test_burning_the_whole_cap_in_the_first_hour_is_paced(self):
        day_start = 10 * DAY_MS
        now = day_start + 3600_000  # 1h into the day
        prior = [day_start + i * 60_000 for i in range(self.CAP)]  # all 30, spread over ~29min
        reason = local_pace_gate(prior, now, self.CAP)
        assert reason is not None and reason.startswith("pace:")

    def test_slack_allows_a_small_burst_right_after_midnight(self):
        day_start = 10 * DAY_MS
        # Immediately at day start, day_frac=0 -> allowed_by_now == slack.
        # With default slack=2, 1 prior submit should still be allowed.
        prior = [day_start + 1]
        assert local_pace_gate(prior, day_start + 2, self.CAP, slack=2.0) is None
        # But a 3rd right away (ahead of the +2 slack budget) is paced.
        prior3 = [day_start + 1, day_start + 2, day_start + 3]
        reason = local_pace_gate(prior3, day_start + 4, self.CAP, slack=2.0)
        assert reason is not None

    def test_resets_on_utc_day_boundary(self):
        day_start = 10 * DAY_MS
        prior = [day_start + i * 60_000 for i in range(self.CAP)]
        next_day_start = 11 * DAY_MS
        assert local_pace_gate(prior, next_day_start, self.CAP) is None

    def test_reason_is_stable_across_repeated_calls_with_same_state(self):
        # Must not embed a live countdown, or it defeats cycle()'s
        # decision-dedup (same reason string every cycle while paced).
        day_start = 10 * DAY_MS
        prior = [day_start + i * 60_000 for i in range(self.CAP)]
        r1 = local_pace_gate(prior, day_start + 3600_000, self.CAP)
        r2 = local_pace_gate(prior, day_start + 3600_000 + 5 * 60_000, self.CAP)
        assert r1 == r2

    def test_zero_cap_never_paces(self):
        assert local_pace_gate([1, 2, 3], 10 * DAY_MS, 0) is None


class TestLocalPairOpenGate:
    """check_pair_open (via hf.open_until_ms) is only enforced from
    HF_OPEN_GATE_FROM — use timestamps well after that."""

    T0 = hf.HF_OPEN_GATE_FROM * 1000 + 10 * DAY_MS
    HORIZON_S = 1800

    def _open_call(self, direction="LONG", entry=100.0, tp_bps=19.0, sl_bps=19.0,
                   t0_ms=None, ticks=None):
        return {
            "direction": direction, "entry": entry, "tp_bps": tp_bps, "sl_bps": sl_bps,
            "t0_ms": t0_ms if t0_ms is not None else self.T0,
            "horizon_s": self.HORIZON_S, "ticks": ticks or [],
        }

    def test_no_prior_calls_is_open(self):
        assert local_pair_open_gate([], self.T0 + 60_000, (self.T0 + 60_000) / 1000) is None

    def test_untouched_call_still_holds_the_pair_until_horizon(self):
        # no ticks at all after t0 -> open_until_ms conservatively returns t_end
        call = self._open_call()
        now = self.T0 + 60_000  # well inside the 30-min horizon
        reason = local_pair_open_gate([call], now, now / 1000)
        assert reason is not None and "pair_open_same_mechanism" in reason

    def test_reentry_allowed_after_a_decisive_touch(self):
        # two ticks that touch TP (100 * 1.0019 = 100.19) satisfy MIN_TOUCH_TICKS=2
        touch_ms = self.T0 + 5_000
        ticks = [
            {"t": touch_ms, "p": 100.20},
            {"t": touch_ms + 1, "p": 100.21},
        ]
        call = self._open_call(ticks=ticks)
        now = touch_ms + 2  # right after the touch closed the call
        assert local_pair_open_gate([call], now, now / 1000) is None

    def test_reentry_allowed_once_horizon_has_fully_elapsed(self):
        call = self._open_call()
        now = self.T0 + self.HORIZON_S * 1000 + 1
        assert local_pair_open_gate([call], now, now / 1000) is None

    def test_a_call_with_no_entry_price_holds_nothing(self):
        call = self._open_call(entry=None)
        now = self.T0 + 60_000
        assert local_pair_open_gate([call], now, now / 1000) is None

    def test_no_op_before_the_open_gate_shipped(self):
        early_t0 = hf.HF_OPEN_GATE_FROM * 1000 - DAY_MS
        call = self._open_call(t0_ms=early_t0)
        now = early_t0 + 60_000
        assert local_pair_open_gate([call], now, now / 1000) is None


class TestLocalCrossLockGate:
    def test_no_lf_history_on_pair_is_unlocked(self):
        assert local_cross_lock_gate("BTCUSD", {}, 10 * DAY_MS) is None

    def test_locked_inside_24h_of_an_lf_call(self):
        lf_ts = 10 * DAY_MS
        now = lf_ts + 3600_000  # 1h later
        reason = local_cross_lock_gate("BTCUSD", {"BTCUSD": lf_ts}, now)
        assert reason is not None and "cross_mechanism_lock:BTCUSD" in reason

    def test_unlocked_once_24h_has_passed(self):
        lf_ts = 10 * DAY_MS
        now = lf_ts + hf.PAIR_LOCK_MS + 1
        assert local_cross_lock_gate("BTCUSD", {"BTCUSD": lf_ts}, now) is None

    def test_only_locks_the_pair_it_was_seen_on(self):
        lf_ts = 10 * DAY_MS
        now = lf_ts + 60_000
        assert local_cross_lock_gate("ETHUSD", {"BTCUSD": lf_ts}, now) is None


class TestEvaluatePair:
    NOW = 10 * DAY_MS

    def _ticks(self, prices_by_age_s: dict[float, float]) -> list[dict]:
        return [{"t": self.NOW - int(age * 1000), "p": p} for age, p in prices_by_age_s.items()]

    def test_not_enough_history_returns_none(self):
        ticks = self._ticks({0: 100.0, 30: 100.1})
        assert evaluate_pair("BTCUSD", ticks, 19.0, 19.0, 1800, self.NOW) is None

    def test_flat_book_scores_no_direction(self):
        ticks = self._ticks({i * 5: 100.0 for i in range(60)})
        res = evaluate_pair("BTCUSD", ticks, 19.0, 19.0, 1800, self.NOW)
        assert res is not None
        assert res["direction"] is None

    def test_sustained_uptrend_scores_long(self):
        # price rises steadily from 300s ago to now -> both fast+slow momentum agree
        ticks = self._ticks({(60 - i) * 5: 99.0 + i * 0.05 for i in range(60)})
        res = evaluate_pair("BTCUSD", ticks, 19.0, 19.0, 1800, self.NOW)
        assert res is not None
        assert res["direction"] == "LONG"
        assert res["score"] > 0

    def test_sustained_downtrend_scores_short(self):
        ticks = self._ticks({(60 - i) * 5: 101.0 - i * 0.05 for i in range(60)})
        res = evaluate_pair("BTCUSD", ticks, 19.0, 19.0, 1800, self.NOW)
        assert res is not None
        assert res["direction"] == "SHORT"
        assert res["score"] > 0

    def test_vanishing_drift_has_no_directional_edge(self):
        # A drift this tiny doesn't even clear MIN_EDGE, so it's gated as
        # "no_edge" before the too_quiet check is ever reached.
        ticks = self._ticks({(60 - i) * 5: 100.0 + i * 0.00001 for i in range(60)})
        res = evaluate_pair("BTCUSD", ticks, 19.0, 19.0, 1800, self.NOW)
        assert res is not None
        assert res["direction"] is None
        assert "no_edge" in res["reason"]

    def test_too_quiet_book_is_gated_even_with_a_directional_drift(self):
        # Shape: rise (age 295->200) -> partial retrace (age 195->65) -> small
        # partial recovery into "now" (age 60->0). Net effect: mom_fast/mom_slow
        # both stay mildly positive (clears MIN_EDGE ~0.06) but price_now sits
        # mid-range rather than at the range high (keeps the mean-reversion
        # guard from firing), while realized_range_bps/tp_bps stays under
        # MIN_VOL_RATIO (0.15) — the gap those two thresholds must leave open
        # for "technically directional but still too quiet" to be reachable
        # at all (see MIN_EDGE's docstring for why diff <= vol_ratio always).
        peak, retrace_frac, fastrise = 0.026, 0.5, 0.5
        prices: dict[float, float] = {}
        for i in range(20):
            prices[295 - i * 5] = 100.0 + (i / 19) * peak
        trough = 100.0 + peak * (1 - retrace_frac)
        for i in range(27):
            prices[195 - i * 5] = (100.0 + peak) - (i / 26) * (peak * retrace_frac)
        for i in range(13):
            prices[60 - i * 5] = trough + (i / 12) * (peak * retrace_frac * fastrise)
        ticks = self._ticks(prices)
        res = evaluate_pair("BTCUSD", ticks, 19.0, 19.0, 1800, self.NOW)
        assert res is not None
        assert res["direction"] is None
        assert "too_quiet" in res["reason"]
        assert "guard" not in res["components"]

    def test_bigger_move_relative_to_band_scores_higher(self):
        # Same shape of move, but XRPUSD's 24 bps band vs a much wider tp_bps
        # below — the smaller-band case should score higher (it's a bigger
        # fraction of ITS OWN band), proving score is continuous/band-relative
        # rather than a fixed step function that pins every agreeing pair alike.
        ticks = self._ticks({(60 - i) * 5: 99.0 + i * 0.05 for i in range(60)})
        tight = evaluate_pair("XRPUSD", ticks, 24.0, 24.0, 1800, self.NOW)
        wide = evaluate_pair("XRPUSD", ticks, 200.0, 200.0, 1800, self.NOW)
        assert tight["direction"] == wide["direction"] == "LONG"
        assert tight["score"] > wide["score"]

    def test_longer_horizon_pair_reads_wider_momentum_windows(self):
        # A price path that reverses inside the 1800s-calibrated fast/slow window
        # but is still net-flat over a 7200s-scaled window should read differently
        # depending on horizon_s, proving the windows actually scale with it.
        ticks = self._ticks({i * 5: 100.0 for i in range(120)})  # flat over 600s
        res_short_horizon = evaluate_pair("TAOUSD", ticks, 53.1, 53.1, 1800, self.NOW)
        res_long_horizon = evaluate_pair("TAOUSD", ticks, 53.1, 53.1, 7200, self.NOW)
        assert res_short_horizon is not None and res_long_horizon is not None
        # Both flat here, but the long-horizon call reads a 4x wider slow window —
        # confirm it didn't just silently ignore horizon_s.
        assert res_long_horizon["direction"] is None

    def test_sustained_trends_do_not_trigger_the_spike_fade(self):
        # A constant-rate ramp has recent_v ~= base_v (accel_ratio ~= 1, well
        # under SPIKE_ACCEL_RATIO=2.5) — the spike fade must stay silent on an
        # ordinary smooth trend, only firing on a genuine acceleration.
        up = self._ticks({(60 - i) * 5: 99.0 + i * 0.05 for i in range(60)})
        down = self._ticks({(60 - i) * 5: 101.0 - i * 0.05 for i in range(60)})
        for ticks, want_dir in ((up, "LONG"), (down, "SHORT")):
            res = evaluate_pair("BTCUSD", ticks, 19.0, 19.0, 1800, self.NOW)
            assert res["direction"] == want_dir
            assert "spike_fade" not in res["components"]


class TestSpikeFade:
    NOW = 10 * DAY_MS

    def _ticks(self, prices_by_age_s: dict[float, float]) -> list[dict]:
        return [{"t": self.NOW - int(age * 1000), "p": p} for age, p in prices_by_age_s.items()]

    def _flat_then_burst(self, burst_ages_to_prices: dict[float, float]) -> list[dict]:
        """A dead-flat book at 100.0 from age 300 down to 20 (57 ticks, still
        the fast/slow momentum windows' baseline), then a fast, concentrated
        burst in the given ages (all < 20s) — the shape both chart examples
        in the user's screenshot show: a sharp, near-vertical move, not a
        gradual ramp."""
        prices = {age: 100.0 for age in range(20, 301, 5)}
        prices.update(burst_ages_to_prices)
        return self._ticks(prices)

    def test_aggressive_up_spike_is_faded_to_short(self):
        # +60 bps crammed into the last 15s against a flat book: recent
        # velocity is far above the fast window's own average rate (which is
        # diluted by 45s of dead flat time before the burst even started).
        ticks = self._flat_then_burst({15: 100.15, 10: 100.3, 5: 100.45, 0: 100.6})
        res = evaluate_pair("BTCUSD", ticks, 19.0, 19.0, 1800, self.NOW)
        assert res is not None
        assert res["direction"] == "SHORT"
        assert res["components"]["spike_fade"] == "short_after_up_spike"
        assert res["components"]["spike_accel"] >= 2.5
        assert res["score"] > 0

    def test_aggressive_down_spike_is_faded_to_long(self):
        ticks = self._flat_then_burst({15: 99.85, 10: 99.7, 5: 99.55, 0: 99.4})
        res = evaluate_pair("BTCUSD", ticks, 19.0, 19.0, 1800, self.NOW)
        assert res is not None
        assert res["direction"] == "LONG"
        assert res["components"]["spike_fade"] == "long_after_down_spike"
        assert res["score"] > 0

    def test_small_wiggle_below_band_floor_is_not_faded(self):
        # Fast relative to its own tiny size, but well under SPIKE_MIN_RATIO
        # (0.25 * 19bps ~= 4.75bps) — a real book will jitter this much on
        # ordinary noise and must not be treated as an exhaustion signal.
        ticks = self._flat_then_burst({15: 100.01, 10: 100.02, 5: 100.03, 0: 100.04})
        res = evaluate_pair("BTCUSD", ticks, 19.0, 19.0, 1800, self.NOW)
        assert res is not None
        assert "spike_fade" not in res["components"]


class TestFastpathWake:
    """should_fastpath_wake is what the tick-reader thread calls on every new
    tick to decide whether to interrupt run_forever()'s EVAL_INTERVAL_S sleep
    early — see hf_auto_bot.HfAutoBot._on_tick."""
    NOW = 10 * DAY_MS

    def _ticks(self, prices_by_age_s: dict[float, float]) -> list[dict]:
        return [{"t": self.NOW - int(age * 1000), "p": p} for age, p in prices_by_age_s.items()]

    def _flat_then_burst(self, burst_ages_to_prices: dict[float, float]) -> list[dict]:
        prices = {age: 100.0 for age in range(20, 301, 5)}
        prices.update(burst_ages_to_prices)
        return self._ticks(prices)

    def test_wakes_on_a_decisive_spike_fade(self):
        ticks = self._flat_then_burst({15: 100.15, 10: 100.3, 5: 100.45, 0: 100.6})
        assert should_fastpath_wake(ticks, 19.0, 19.0, 1800, self.NOW, min_score=0.0) is True

    def test_does_not_wake_below_the_score_floor(self):
        # Same decisive spike as above, but a floor no real score can clear —
        # proves min_score is actually enforced, not just decorative.
        ticks = self._flat_then_burst({15: 100.15, 10: 100.3, 5: 100.45, 0: 100.6})
        assert should_fastpath_wake(ticks, 19.0, 19.0, 1800, self.NOW, min_score=0.999) is False

    def test_does_not_wake_on_a_sustained_trend_without_a_spike(self):
        # Ordinary smooth momentum, no burst -> evaluate_pair calls a
        # direction but never sets spike_fade (see
        # test_sustained_trends_do_not_trigger_the_spike_fade above).
        ticks = self._ticks({(60 - i) * 5: 99.0 + i * 0.05 for i in range(60)})
        assert should_fastpath_wake(ticks, 19.0, 19.0, 1800, self.NOW, min_score=0.0) is False

    def test_does_not_wake_with_too_little_history(self):
        ticks = self._ticks({0: 100.0, 30: 100.1})
        assert should_fastpath_wake(ticks, 19.0, 19.0, 1800, self.NOW, min_score=0.0) is False


class TestRotationPairFor:
    """rotation_pair_for is what restricts the bot to ONE pair per UTC day
    (user request 2026-09-01) — deterministic on epoch day, no persisted
    start-date."""
    DAY_S = 86_400

    def test_cycles_through_the_pool_by_utc_day(self):
        pairs = ["XRPUSD", "BTCUSD", "TAOUSD"]
        picks = {rotation_pair_for(pairs, day * self.DAY_S) for day in range(3)}
        # Three consecutive days must hit all three pairs, not repeat one —
        # otherwise this isn't rotating at all.
        assert picks == set(pairs)

    def test_same_utc_day_always_picks_the_same_pair(self):
        pairs = ["XRPUSD", "BTCUSD", "TAOUSD"]
        start_of_day = 5 * self.DAY_S
        a = rotation_pair_for(pairs, start_of_day)
        b = rotation_pair_for(pairs, start_of_day + self.DAY_S - 1)
        assert a == b

    def test_day_boundary_flips_the_pick(self):
        pairs = ["XRPUSD", "BTCUSD", "TAOUSD"]
        end_of_day = 5 * self.DAY_S - 1
        start_of_next = 5 * self.DAY_S
        assert rotation_pair_for(pairs, end_of_day) != rotation_pair_for(pairs, start_of_next)

    def test_falls_back_to_tradeable_pairs_when_none_of_the_order_is_live(self):
        # ROTATION_ORDER's default pool shares nothing with what's actually
        # tradeable right now (e.g. a narrower PAIR_ALLOWLIST) -> rotate
        # through what IS tradeable instead of returning nothing.
        assert rotation_pair_for(["ETHUSD"], 0) == "ETHUSD"

    def test_single_tradeable_pair_always_wins(self):
        for day in range(5):
            assert rotation_pair_for(["BTCUSD"], day * self.DAY_S) == "BTCUSD"

    def test_no_tradeable_pairs_returns_none(self):
        assert rotation_pair_for([], 0) is None


class TestCallJustSealed:
    """call_just_sealed is what lets the bot react within seconds of an open
    call resolving (user request 2026-09-01) instead of sitting on a freed-up
    pair until the next scheduled cycle — see HfAutoBot._check_seal_wake."""
    T0_MS = 10 * DAY_MS

    def test_not_sealed_while_price_sits_between_tp_and_sl(self):
        ticks = [{"t": self.T0_MS + i * 1000, "p": 100.0} for i in range(5)]
        sealed = call_just_sealed("LONG", 100.0, 19.0, 19.0, self.T0_MS, 1800,
                                  ticks, self.T0_MS + 5000)
        assert sealed is False

    def test_sealed_once_tp_is_touched(self):
        # MIN_TOUCH_TICKS=2: a single tick past TP is a wick, not a touch —
        # needs two ticks at/above it to seal (mirrors hf.open_until_ms).
        ticks = [{"t": self.T0_MS + 1000, "p": 100.0},
                 {"t": self.T0_MS + 2000, "p": 100.25},   # +25bps clears a 19bps TP
                 {"t": self.T0_MS + 3000, "p": 100.26}]
        sealed = call_just_sealed("LONG", 100.0, 19.0, 19.0, self.T0_MS, 1800,
                                  ticks, self.T0_MS + 3000)
        assert sealed is True

    def test_sealed_once_the_horizon_washes_out(self):
        ticks = [{"t": self.T0_MS + 1000, "p": 100.0}]
        horizon_s = 1800
        before_end = self.T0_MS + horizon_s * 1000 - 1
        after_end = self.T0_MS + horizon_s * 1000 + 1
        assert call_just_sealed("LONG", 100.0, 19.0, 19.0, self.T0_MS, horizon_s,
                                ticks, before_end) is False
        assert call_just_sealed("LONG", 100.0, 19.0, 19.0, self.T0_MS, horizon_s,
                                ticks, after_end) is True

    def test_a_void_call_with_no_entry_reads_as_already_sealed(self):
        # Mirrors hf.open_until_ms: no entry price -> holds nothing -> t0_ms,
        # which is always <= any now_ms >= t0_ms.
        assert call_just_sealed("LONG", None, 19.0, 19.0, self.T0_MS, 1800,
                                [], self.T0_MS) is True


class TestConfirmStreak:
    def test_first_direction_starts_streak_at_one(self):
        state, n = confirm_streak(None, "LONG")
        assert n == 1 and state == {"direction": "LONG", "n": 1}

    def test_same_direction_increments(self):
        state, _ = confirm_streak(None, "LONG")
        state, n = confirm_streak(state, "LONG")
        assert n == 2

    def test_direction_change_resets_to_one(self):
        state, _ = confirm_streak(None, "LONG")
        state, _ = confirm_streak(state, "LONG")
        state, n = confirm_streak(state, "SHORT")
        assert n == 1

    def test_none_direction_resets_state(self):
        state, _ = confirm_streak(None, "LONG")
        state, n = confirm_streak(state, None)
        assert n == 0 and state is None


class TestParseManualRequest:
    """Validates the dashboard's manual-submit button (user request
    2026-08-31): a request must name a real pair on the LIVE board and a
    real direction, or the bot should drop it rather than attempt garbage."""
    BOARD = {"BTCUSD": (19.0, 19.0, 1800, "crypto"), "XRPUSD": (19.0, 19.0, 1800, "crypto")}

    def test_valid_request_parses(self):
        assert parse_manual_request({"pair": "btcusd", "direction": "long"}, self.BOARD) \
            == ("BTCUSD", "LONG")

    def test_pair_not_on_board_rejected(self):
        assert parse_manual_request({"pair": "TAOUSD", "direction": "LONG"}, self.BOARD) is None

    def test_bad_direction_rejected(self):
        assert parse_manual_request({"pair": "BTCUSD", "direction": "UP"}, self.BOARD) is None

    def test_missing_fields_rejected(self):
        assert parse_manual_request({}, self.BOARD) is None


class TestFailedBreakoutReversal:
    """EXPERIMENTAL (user request 2026-08-31): a wash that got CLOSE to TP
    (>= mfe_ratio of the band) without ever seriously testing the SL
    (<= mae_ratio of the band) reads as a failed breakout — reverse it. A
    two-way whipsaw wash, or any decisive (non-wash) call, should NOT."""
    T0_MS = 10 * DAY_MS
    ENTRY = 100.0
    TP_BPS = SL_BPS = 19.0
    HORIZON_S = 1800

    def _ticks(self, prices, start_offset_ms=1000, step_ms=1000):
        return [{"t": self.T0_MS + start_offset_ms + i * step_ms, "p": p}
                for i, p in enumerate(prices)]

    def test_clean_near_miss_long_triggers_short_reversal(self):
        # +17bps (89% of the 19bps TP) without ever dipping past -9.5bps (50%
        # of the 19bps SL) — a clean, one-directional attempt that stalled.
        ticks = self._ticks([100.0, 100.17, 100.05, 99.97, 100.10])
        result = failed_breakout_reversal("LONG", self.ENTRY, self.TP_BPS, self.SL_BPS,
                                          self.T0_MS, self.HORIZON_S, ticks)
        assert result == "SHORT"

    def test_clean_near_miss_short_triggers_long_reversal(self):
        ticks = self._ticks([100.0, 99.83, 99.95, 100.03, 99.90])
        result = failed_breakout_reversal("SHORT", self.ENTRY, self.TP_BPS, self.SL_BPS,
                                          self.T0_MS, self.HORIZON_S, ticks)
        assert result == "LONG"

    def test_mfe_too_shallow_no_reversal(self):
        # Only +8bps (42% of TP) — never got close enough to call it a
        # "failed breakout" in the first place.
        ticks = self._ticks([100.0, 100.08, 100.02, 99.97, 100.03])
        result = failed_breakout_reversal("LONG", self.ENTRY, self.TP_BPS, self.SL_BPS,
                                          self.T0_MS, self.HORIZON_S, ticks)
        assert result is None

    def test_two_way_whipsaw_no_reversal(self):
        # Got close to TP (+17bps) but ALSO deep into SL territory (-17bps) —
        # this is chop, not a directional failed-breakout read.
        ticks = self._ticks([100.0, 100.17, 99.83, 100.05, 99.90])
        result = failed_breakout_reversal("LONG", self.ENTRY, self.TP_BPS, self.SL_BPS,
                                          self.T0_MS, self.HORIZON_S, ticks)
        assert result is None

    def test_decisive_win_no_reversal(self):
        # Actually touches TP (with >= MIN_TOUCH_TICKS=2 ticks) — this seals
        # as a WIN, not a wash, so the reversal logic must not apply at all.
        ticks = self._ticks([100.0, 100.20, 100.21])
        result = failed_breakout_reversal("LONG", self.ENTRY, self.TP_BPS, self.SL_BPS,
                                          self.T0_MS, self.HORIZON_S, ticks)
        assert result is None

    def test_no_entry_price_no_reversal(self):
        ticks = self._ticks([100.0, 100.17])
        assert failed_breakout_reversal("LONG", None, self.TP_BPS, self.SL_BPS,
                                        self.T0_MS, self.HORIZON_S, ticks) is None
