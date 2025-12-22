from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd

TRADING_DAYS_DEFAULT = 252


# -----------------------
# Data contracts
# -----------------------
# prices: pd.DataFrame with DatetimeIndex and column "price" (float)


@dataclass(frozen=True)
class BacktestConfig:
    start_capital: float
    monthly_contribution: float
    contrib_day: int  # 1-28
    slippage_bps: float = 0.0
    trading_days: int = TRADING_DAYS_DEFAULT


@dataclass
class BacktestResult:
    equity: pd.Series
    contrib: pd.Series
    shares: pd.Series
    summary: dict


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        raise ValueError("Preisdaten sind leer.")
    if "price" not in prices.columns:
        raise ValueError('Preisdaten müssen eine Spalte "price" enthalten.')
    px = prices["price"].astype(float).dropna().to_frame("price")
    px.index = pd.to_datetime(px.index)
    if px.empty:
        raise ValueError("Preisdaten enthalten nach dem Bereinigen keine Werte.")
    return px


def build_contrib_dates(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp, day: int) -> pd.DatetimeIndex:
    """Planned monthly contribution dates mapped to the next available trading day in index."""
    if day < 1 or day > 28:
        raise ValueError("contrib_day muss zwischen 1 und 28 liegen.")

    months = pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M")
    planned = []
    for p in months:
        last_day = (pd.Timestamp(p.year, p.month, 1) + pd.offsets.MonthEnd(0)).day
        planned.append(pd.Timestamp(p.year, p.month, min(day, last_day)))

    contrib = []
    for d in planned:
        ix = index.searchsorted(d)
        if ix < len(index):
            contrib.append(index[ix])

    contrib = [d for d in contrib if (d >= index[0]) and (d <= index[-1])]
    return pd.DatetimeIndex(sorted(set(contrib)))


def run_dca_backtest(prices: pd.DataFrame, cfg: BacktestConfig, start: pd.Timestamp, end: pd.Timestamp) -> BacktestResult:
    """
    DCA backtest:
    - invest start_capital on first day
    - invest monthly_contribution on contrib dates (next trading day)
    - slippage applies to buy prices only
    Returns equity (portfolio value), contrib (cum invested), shares, summary.
    """
    prices = _validate_prices(prices)
    px = prices.loc[start:end, "price"].astype(float).dropna()
    if px.empty:
        raise ValueError("Keine Preisdaten im gewählten Zeitraum.")

    idx = px.index
    contrib_dates = build_contrib_dates(idx, start, end, int(cfg.contrib_day))

    shares = pd.Series(index=idx, dtype="float64")

    # Initial buy
    first_px = float(px.iloc[0])
    shares.iloc[0] = (cfg.start_capital / first_px) if first_px > 0 else 0.0

    # DCA buys (sparse updates, then forward-fill)
    for d in contrib_dates:
        buy_px = float(px.loc[d]) * (1.0 + cfg.slippage_bps / 10_000.0)
        if buy_px <= 0 or cfg.monthly_contribution <= 0:
            continue
        prev = float(shares.loc[:d].ffill().iloc[-1])
        shares.loc[d] = prev + (cfg.monthly_contribution / buy_px)

    shares = shares.ffill().fillna(0.0)
    equity = shares * px

    # Contributions (deterministic for given dates)
    contrib = pd.Series(0.0, index=idx)
    contrib.iloc[0] = float(cfg.start_capital)
    for d in contrib_dates:
        contrib.loc[d] += float(cfg.monthly_contribution)
    contrib = contrib.cumsum()

    final_value = float(equity.iloc[-1])
    invested = float(contrib.iloc[-1])
    pl_abs = final_value - invested
    pl_pct = (final_value / invested - 1.0) * 100.0 if invested > 0 else np.nan

    summary = {
        "final_value": final_value,
        "invested": invested,
        "pl_abs": pl_abs,
        "pl_pct": pl_pct,
        "start": str(idx[0].date()),
        "end": str(idx[-1].date()),
        "n_days": int(len(idx)),
    }
    return BacktestResult(equity=equity, contrib=contrib, shares=shares, summary=summary)


def make_synthetic_leverage_prices(
    base_prices: pd.DataFrame,
    leverage: float,
    annual_fin: float,
    annual_fee: float,
    trading_days: int = TRADING_DAYS_DEFAULT,
) -> pd.DataFrame:
    """
    Synthetic daily-reset leveraged price series:
      lev_ret = leverage * base_ret - daily_fin - daily_fee
    with clipping at -95% daily return to avoid negative prices.
    """
    base_prices = _validate_prices(base_prices)
    p = base_prices["price"].astype(float).dropna()

    rets = p.pct_change().fillna(0.0)
    daily_fin = (1.0 + float(annual_fin)) ** (1.0 / trading_days) - 1.0
    daily_fee = (1.0 + float(annual_fee)) ** (1.0 / trading_days) - 1.0

    lev_rets = float(leverage) * rets - daily_fin - daily_fee
    lev_rets = lev_rets.clip(lower=-0.95)

    lev_price = (1.0 + lev_rets).cumprod() * float(p.iloc[0])
    return pd.DataFrame({"price": lev_price}, index=p.index)
