"""Track record scorer — offline, with prices and saved runs mocked."""
from datetime import datetime
from unittest.mock import patch

import pandas as pd

from analytics.track_record import evaluate

# Four business days of closes for one ticker.
_DATES = pd.to_datetime(["2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13"])
_CLOSES = pd.Series([100.0, 110.0, 99.0, 99.0], index=_DATES)


def _run(**over):
    base = {
        "id": 1, "created_at": datetime(2026, 8, 10, 16, 0), "ticker": "TEST",
        "company": "Test Co", "action": "Buy", "confidence": 0.6,
        "last_close": 100.0, "next_day_price": 104.0,   # predicts +4%
        "beats_baseline": True,
    }
    base.update(over)
    return base


def _evaluate(runs, closes=_CLOSES):
    with patch("analytics.track_record.all_runs", return_value=runs), \
         patch("analytics.track_record.fetch_prices",
               return_value=pd.DataFrame({"Close": closes})):
        return evaluate()


def test_correct_up_call_is_scored_as_a_hit():
    rec = _evaluate([_run()])                      # predicted +4%, actual 100 -> 110
    assert rec.n_graded == 1 and rec.n_correct == 1
    assert rec.hit_rate == 1.0
    g = rec.graded[0]
    assert g.direction_correct
    assert round(g.actual_return, 4) == 0.10
    assert round(g.predicted_return, 4) == 0.04
    assert round(g.abs_pct_error, 4) == round(abs(104.0 - 110.0) / 110.0, 4)


def test_wrong_direction_is_scored_as_a_miss():
    # Made on the 11th (close 110), predicts up; actual falls to 99.
    rec = _evaluate([_run(created_at=datetime(2026, 8, 11, 16, 0),
                          last_close=110.0, next_day_price=114.0)])
    assert rec.n_graded == 1 and rec.n_correct == 0
    assert rec.hit_rate == 0.0
    assert not rec.graded[0].direction_correct


def test_run_on_the_last_bar_is_pending_not_a_miss():
    """The next trading day hasn't closed — that must not count against the hit rate."""
    rec = _evaluate([_run(created_at=datetime(2026, 8, 13, 16, 0), last_close=99.0)])
    assert rec.pending == 1
    assert rec.n_graded == 0 and rec.hit_rate is None


def test_run_that_cannot_be_matched_to_a_bar_is_unverifiable():
    """last_close matches no bar, so we must not guess which one it came from."""
    rec = _evaluate([_run(last_close=555.55)])
    assert rec.unverifiable == 1
    assert rec.n_graded == 0


def test_missing_prices_do_not_raise():
    with patch("analytics.track_record.all_runs", return_value=[_run()]), \
         patch("analytics.track_record.fetch_prices", side_effect=RuntimeError("no data")):
        rec = evaluate()
    assert rec.unverifiable == 1 and rec.n_graded == 0


def test_hit_rate_and_error_average_over_several_runs():
    runs = [
        _run(),                                                     # hit
        _run(id=2, created_at=datetime(2026, 8, 11, 16, 0),
             last_close=110.0, next_day_price=114.0),               # miss
    ]
    rec = _evaluate(runs)
    assert rec.n_graded == 2 and rec.n_correct == 1
    assert rec.hit_rate == 0.5
    assert rec.mean_abs_pct_error > 0


def test_small_sample_is_flagged_honestly():
    rec = _evaluate([_run()])
    assert any("too few" in n for n in rec.notes)


def test_no_runs_reports_nothing_rather_than_zero_percent():
    rec = _evaluate([])
    assert rec.total_runs == 0 and rec.hit_rate is None
    assert rec.notes
