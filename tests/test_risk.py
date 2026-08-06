import numpy as np
import pandas as pd

from analytics.risk import compute_risk


def _prices(close: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=len(close), freq="B")
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                         "Volume": 1_000_000.0}, index=idx)


def test_monotonic_up_has_no_drawdown():
    rk = compute_risk(_prices([100 * (1.001 ** i) for i in range(300)]))
    assert rk.max_drawdown >= -1e-6          # essentially zero
    assert rk.annual_volatility >= 0.0


def test_known_drawdown():
    # up to 120 then down to 60 → drawdown of -50%
    close = list(np.linspace(100, 120, 50)) + list(np.linspace(120, 60, 50))
    rk = compute_risk(_prices(close))
    assert abs(rk.max_drawdown - (-0.5)) < 0.02
    assert rk.drawdown_peak is not None and rk.drawdown_trough is not None


def test_flat_price_zero_vol_none_sharpe():
    rk = compute_risk(_prices([100.0] * 200))
    assert rk.annual_volatility == 0.0
    assert rk.sharpe_like is None            # guarded against div-by-zero


def test_beta_vs_itself_is_one():
    rng = np.random.default_rng(1)
    close = list(100 * np.exp(np.cumsum(rng.normal(0, 0.012, 260))))
    px = _prices(close)
    rk = compute_risk(px, benchmark_prices=px, benchmark_name="SELF")
    assert rk.beta is not None and abs(rk.beta - 1.0) < 1e-6


def test_no_benchmark_gives_none_beta():
    rng = np.random.default_rng(2)
    close = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    rk = compute_risk(_prices(close))
    assert rk.beta is None


def test_biggest_moves_ordered_and_var_signs():
    rng = np.random.default_rng(3)
    close = list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    rk = compute_risk(_prices(close))
    assert rk.biggest_up and rk.biggest_up[0].pct >= rk.biggest_up[-1].pct
    assert rk.biggest_down and rk.biggest_down[0].pct <= rk.biggest_down[-1].pct
    assert rk.var_95 <= 0.0 and rk.var_99 <= rk.var_95   # 99% tail no less extreme than 95%
