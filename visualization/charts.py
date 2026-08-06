"""Plotly chart builders for the dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from orchestration.schemas import CompareItem, ForecastResult, RiskProfile, SentimentSummary
from technical_analysis.indicators import compute_indicators
from visualization import theme  # noqa: F401  (registers the shared Plotly template)

_ACTUAL = "#2563eb"
_PRED = "#dc2626"


def price_and_forecast(prices: pd.DataFrame, forecast: ForecastResult, lookback: int = 120) -> go.Figure:
    """Recent close line + forward horizon prediction markers."""
    df = prices.tail(lookback)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Actual Close",
                             line=dict(color=_ACTUAL, width=2)))
    last_date = df.index[-1]
    for h in forecast.ensemble.horizons:
        fut = last_date + pd.tseries.offsets.BusinessDay(h.horizon_days)
        fig.add_trace(go.Scatter(
            x=[last_date, fut], y=[forecast.last_close, h.predicted_price],
            name=f"{h.horizon} → ${h.predicted_price:.2f}", mode="lines+markers",
            line=dict(color=_PRED, width=1.5, dash="dot"),
        ))
    fig.update_layout(title="Price & Forward Forecast", height=420,
                      hovermode="x unified", margin=dict(t=50, b=20))
    fig.update_yaxes(title_text="Price ($)")
    return fig


def backtest_actual_vs_pred(forecast: ForecastResult) -> go.Figure:
    """The signature 'actual vs predicted' chart over the held-out test window."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast.backtest_dates, y=forecast.backtest_actual,
                             name="Actual Price", line=dict(color=_ACTUAL, width=2)))
    fig.add_trace(go.Scatter(x=forecast.backtest_dates, y=forecast.backtest_pred,
                             name="Predicted Price", line=dict(color=_PRED, width=2, dash="dash")))
    fig.update_layout(title="Holdout: Actual vs Predicted (next-day reconstruction)",
                      height=380, hovermode="x unified", margin=dict(t=50, b=20))
    fig.update_yaxes(title_text="Price ($)")
    return fig


def technical_chart(prices: pd.DataFrame, lookback: int = 180) -> go.Figure:
    """Candlesticks + SMA/EMA/Bollinger, with RSI and MACD sub-panels."""
    ind = compute_indicators(prices).tail(lookback)
    df = prices.tail(lookback)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.04,
                        subplot_titles=("Price · SMA/EMA · Bollinger", "RSI (14)", "MACD"))

    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                 close=df["Close"], name="OHLC", showlegend=False), row=1, col=1)
    for col, color, name in [("sma_20", "#f59e0b", "SMA 20"), ("sma_50", "#8b5cf6", "SMA 50"),
                             ("ema_12", "#10b981", "EMA 12")]:
        fig.add_trace(go.Scatter(x=ind.index, y=ind[col], name=name, line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["bb_high"], name="BB High",
                             line=dict(width=1, color="#94a3b8", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["bb_low"], name="BB Low", fill="tonexty",
                             fillcolor="rgba(148,163,184,0.1)",
                             line=dict(width=1, color="#94a3b8", dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=ind.index, y=ind["rsi_14"], name="RSI",
                             line=dict(color="#2563eb", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#dc2626", dash="dash", width=1), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#16a34a", dash="dash", width=1), row=2, col=1)

    fig.add_trace(go.Scatter(x=ind.index, y=ind["macd"], name="MACD",
                             line=dict(color="#2563eb", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=ind.index, y=ind["macd_signal"], name="Signal",
                             line=dict(color="#f59e0b", width=1.5)), row=3, col=1)
    fig.add_trace(go.Bar(x=ind.index, y=ind["macd_diff"], name="Hist",
                         marker_color="#94a3b8"), row=3, col=1)

    fig.update_layout(height=720, hovermode="x unified", margin=dict(t=40, b=20),
                      xaxis_rangeslider_visible=False)
    return fig


def sentiment_gauge(summary: SentimentSummary) -> go.Figure:
    """A −1..+1 gauge of the credibility-weighted news sentiment."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=summary.weighted_score,
        number=dict(valueformat="+.2f"),
        title=dict(text=f"News Sentiment · {summary.label}"),
        gauge=dict(
            axis=dict(range=[-1, 1]),
            bar=dict(color="#1f2937"),
            steps=[
                dict(range=[-1, -0.05], color="#fecaca"),
                dict(range=[-0.05, 0.05], color="#e5e7eb"),
                dict(range=[0.05, 1], color="#bbf7d0"),
            ],
        ),
    ))
    fig.update_layout(height=280, margin=dict(t=50, b=20))
    return fig


def feature_importance_bar(forecast: ForecastResult) -> go.Figure:
    items = list(forecast.feature_importance.items())[:10][::-1]
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
        marker_color="#2563eb",
    ))
    fig.update_layout(title="Top Feature Importances (ensemble)", height=360,
                      margin=dict(t=50, b=20), xaxis_title="relative importance")
    return fig


def drawdown_chart(risk: RiskProfile) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=risk.drawdown_dates, y=[v * 100 for v in risk.drawdown_series],
        fill="tozeroy", name="Drawdown", line=dict(color=_PRED, width=1),
    ))
    fig.update_layout(title="Drawdown from running peak (%)", height=320,
                      hovermode="x unified", margin=dict(t=50, b=20))
    fig.update_yaxes(title_text="Drawdown %")
    return fig


def rolling_vol_chart(risk: RiskProfile) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=risk.rolling_vol_dates, y=[v * 100 for v in risk.rolling_vol],
        name="Annualized volatility", line=dict(color=_ACTUAL, width=2),
    ))
    fig.update_layout(title="21-day rolling annualized volatility (%)", height=320,
                      hovermode="x unified", margin=dict(t=50, b=20))
    fig.update_yaxes(title_text="Volatility %")
    return fig


def compare_prices_chart(items: list[CompareItem]) -> go.Figure:
    """Overlay rebased-to-100 price series for the compared tickers."""
    fig = go.Figure()
    for it in items:
        if it.rebased:
            fig.add_trace(go.Scatter(x=it.dates, y=it.rebased, name=it.ticker, mode="lines"))
    fig.update_layout(title="Rebased price (start = 100)", height=400,
                      hovermode="x unified", margin=dict(t=50, b=20))
    fig.update_yaxes(title_text="Rebased")
    return fig
