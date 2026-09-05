"""bots/dashboard.py — rolling-emission-window summary rollup.

The panel's job is to show the window the validator actually pays on, so these
tests pin the two properties that make it honest: the window boundary and the
linear curve match scoring.decayed_qwin_tally, and a win earned while below the
qualify gate is reported as a raw win but NOT as a qualified one.

They deliberately do NOT assert the rollup equals decayed_qwin_tally — it can't,
since the per-win tier and wash-efficiency multipliers are absent from the call
feed. What is asserted is that the shape is the same.
"""
import pytest

from bots.dashboard import _weekly_rollup
from sn89_signals import config, scoring

DAY = 86_400
NOW = 1_760_000_000.0
WEEK = config.EMISSION_DECAY_S


def call(age_s: float, status: str) -> dict:
    return {"t0_unix": NOW - age_s, "status": status}


def roll(calls, *, win_cap=config.WIN_CAP, eligible=True, min_decisive=8):
    return _weekly_rollup(
        calls, NOW,
        decay_s=WEEK, win_cap=win_cap,
        min_decisive=min_decisive, lb_floor_pct=50.0, eligible=eligible,
    )


def clean_record(n: int, spacing_s: float = 3600.0, status: str = "won") -> list[dict]:
    """n outcomes, newest first, tight enough to sit inside the emission window."""
    return [call(i * spacing_s, status) for i in range(n)]


class TestWindow:
    def test_win_older_than_the_window_is_invisible(self):
        w = roll([call(WEEK + DAY, "won")])
        assert w["won"] == 0
        assert w["decay_sum"] == 0.0

    def test_win_exactly_at_the_boundary_is_excluded(self):
        # decayed_qwin_tally uses `now - t0 < W`, so the boundary is open.
        assert roll([call(WEEK, "won")])["won"] == 0
        assert roll([call(WEEK - 1, "won")])["won"] == 1

    def test_future_dated_call_is_ignored(self):
        assert roll([call(-DAY, "won")])["won"] == 0

    def test_calls_without_a_timestamp_are_skipped(self):
        assert roll([{"t0_unix": 0, "status": "won"}])["won"] == 0

    def test_losses_and_washes_counted_only_inside_the_window(self):
        w = roll([call(DAY, "lost"), call(DAY, "washed"),
                  call(WEEK + DAY, "lost"), call(WEEK + DAY, "washed")])
        assert (w["lost"], w["washed"]) == (1, 1)


class TestDecayCurve:
    def test_fresh_win_is_worth_full_weight(self):
        assert roll([call(0, "won")])["decay_sum"] == pytest.approx(1.0)

    def test_half_aged_win_is_worth_half(self):
        assert roll([call(WEEK / 2, "won")])["decay_sum"] == pytest.approx(0.5)

    def test_curve_matches_the_validator_at_unit_weight(self):
        """Same linear ramp as scoring.decayed_qwin_tally when every tier
        weight is 1.0 — the proxy differs only by the multiplier."""
        ages = [0.0, DAY, 3 * DAY, 6 * DAY]
        got = roll([call(a, "won") for a in ages])["decay_sum"]
        want = scoring.decayed_qwin_tally([(NOW - a, 1.0) for a in ages], NOW)
        assert got == pytest.approx(want, abs=1e-3)

    def test_decay_is_monotone_in_age(self):
        fresh = roll([call(DAY, "won")])["decay_sum"]
        stale = roll([call(5 * DAY, "won")])["decay_sum"]
        assert fresh > stale > 0


class TestWinCap:
    def test_only_the_most_recent_cap_wins_count_toward_the_sum(self):
        w = roll(clean_record(5), win_cap=2)
        # newest two are at age 0 and 1h
        want = (1.0 - 0.0 / WEEK) + (1.0 - 3600.0 / WEEK)
        assert w["decay_sum"] == pytest.approx(want, abs=1e-3)
        assert w["cap_binding"] is True

    def test_raw_count_still_reports_every_live_win(self):
        # the cap bounds the SUM, not the count — a miner should still see
        # what they actually did this week
        assert roll(clean_record(5), win_cap=2)["won"] == 5

    def test_cap_not_flagged_when_it_does_not_bind(self):
        assert roll(clean_record(3), win_cap=20)["cap_binding"] is False


class TestQualification:
    def test_thin_sample_wins_are_raw_but_not_qualified(self):
        # 3 decisive < QUALIFY_MIN_DECISIVE: nothing can qualify yet
        w = roll(clean_record(3))
        assert w["won"] == 3
        assert w["qualified_won"] == 0
        assert w["decay_sum"] > 0
        assert w["decay_sum_qualified"] == 0.0

    def test_coin_flip_record_fails_the_wilson_floor(self):
        # 4W/4L → Wilson LB ≈ 29% at z=1.2816, well under the 50% floor
        calls = [call(i * 3600.0, "won" if i % 2 else "lost") for i in range(8)]
        w = roll(calls)
        assert w["won"] == 4
        assert w["qualified_won"] == 0

    def test_strong_record_qualifies_its_wins(self):
        w = roll(clean_record(12))
        assert w["won"] == 12
        # the earliest wins sit under the 8-decisive floor, later ones clear it
        assert 0 < w["qualified_won"] < 12
        assert w["decay_sum_qualified"] <= w["decay_sum"]

    def test_qualified_sum_never_exceeds_raw_sum(self):
        w = roll(clean_record(30))
        assert w["decay_sum_qualified"] <= w["decay_sum"]

    def test_matches_scoring_qualified_wins_on_a_clean_record(self):
        """Cross-check the as-of gate against the validator's own routine."""
        calls = clean_record(20)
        w = roll(calls)
        decisive = sorted(
            [(float(c["t0_unix"]), c["status"] == "won", False) for c in calls],
            key=lambda d: d[0])
        # first_seen far enough back that warmup never binds
        qwins = scoring.qualified_wins(decisive, NOW - 400 * DAY)
        assert w["qualified_won"] == len(qwins)


class TestEligibility:
    def test_ineligible_miner_earns_nothing(self):
        w = roll(clean_record(20), eligible=False)
        assert w["qualified_won"] == 0
        assert w["decay_sum_qualified"] == 0.0
        assert w["eligible"] is False
        # raw activity is still reported — the miner did trade
        assert w["won"] == 20


class TestReporting:
    def test_approx_flag_set_when_history_is_short(self):
        assert roll(clean_record(10))["qualified_approx"] is True

    def test_approx_flag_clear_with_a_full_reputation_window_behind_it(self):
        calls = clean_record(10) + [call(config.HIT_RATE_WINDOW_S + 10 * DAY, "won")]
        assert roll(calls)["qualified_approx"] is False

    def test_newest_win_age_reported(self):
        assert roll([call(2 * 3600.0, "won")])["newest_win_age_h"] == pytest.approx(2.0)

    def test_no_wins_reports_no_age(self):
        assert roll([call(DAY, "lost")])["newest_win_age_h"] is None

    def test_window_days_is_the_emission_horizon(self):
        assert roll([])["window_days"] == pytest.approx(WEEK / DAY)

    def test_empty_ledger_is_all_zero_not_an_error(self):
        w = roll([])
        assert (w["won"], w["lost"], w["washed"]) == (0, 0, 0)
        assert w["decay_sum"] == 0.0
        assert w["qualified_approx"] is False
