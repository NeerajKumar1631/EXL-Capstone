from unittest.mock import patch

from orchestration.schemas import ScoreCard
from screener import screener


def _fake_score(ticker: str, region=None) -> ScoreCard:
    comp = float(sum(ord(c) for c in ticker) % 100)
    return ScoreCard(ticker=ticker, composite=comp, momentum=comp, trend=comp, low_vol=comp)


def test_screen_ranks_desc_and_reports_cap():
    with patch("screener.screener.quick_score", side_effect=_fake_score):
        lb = screener.screen("US", "dow30", limit=10)
    assert lb.scored == 10 and lb.failed == 0
    comps = [c.composite for c in lb.cards]
    assert comps == sorted(comps, reverse=True)          # ranked high→low
    assert lb.capped is True                             # dow30 (30) > 10
    assert lb.requested == 10 and lb.total_constituents == 30


def test_screen_tolerates_failures_and_reports_coverage():
    def maybe_fail(ticker: str, region=None) -> ScoreCard:
        if ticker in ("AAPL", "AMZN"):
            raise ValueError("boom")
        return ScoreCard(ticker=ticker, composite=50.0)

    with patch("screener.screener.quick_score", side_effect=maybe_fail):
        lb = screener.screen("US", "dow30", limit=6)     # first 6: AAPL,AMGN,AMZN,AXP,BA,CAT
    assert lb.failed == 2                                # AAPL + AMZN
    assert lb.scored == 4
    assert lb.scored + lb.failed == lb.requested == 6


def test_screen_all_fail_gives_empty_leaderboard():
    with patch("screener.screener.quick_score", side_effect=ValueError("nope")):
        lb = screener.screen("INDIA", "niftybank", limit=5)
    assert lb.scored == 0 and lb.failed == 5 and lb.cards == []
