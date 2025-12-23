from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
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


@dataclass(frozen=True)
class MonteCarloConfig:
    horizon_years: int
    n_paths: int
    mode: str  # "start_und_sparplan" | "nur_sparplan" | "nur_start"


@dataclass
class MonteCarloResult:
    summary: pd.DataFrame
    end_values: dict          # {"base": np.ndarray, "syn": np.ndarray}
    bands: dict               # {"base": {"p5": Series, ...}, "syn": {...}}
    index: pd.DatetimeIndex   # future index


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


# -----------------------
# Backtesting
# -----------------------
def run_dca_backtest(prices: pd.DataFrame, cfg: BacktestConfig, start: pd.Timestamp, end: pd.Timestamp) -> BacktestResult:
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
    base_prices = _validate_prices(base_prices)
    p = base_prices["price"].astype(float).dropna()
    rets = p.pct_change().fillna(0.0)

    daily_fin = (1.0 + float(annual_fin)) ** (1.0 / trading_days) - 1.0
    daily_fee = (1.0 + float(annual_fee)) ** (1.0 / trading_days) - 1.0

    lev_rets = float(leverage) * rets - daily_fin - daily_fee
    lev_rets = lev_rets.clip(lower=-0.95)

    lev_price = (1.0 + lev_rets).cumprod() * float(p.iloc[0])
    return pd.DataFrame({"price": lev_price}, index=p.index)


# -----------------------
# Monte Carlo (Bootstrapping)
# -----------------------
def _dca_equity_path(
    prices: pd.Series,
    cfg: BacktestConfig,
    mode: str,
) -> pd.Series:
    """
    Daily portfolio value over provided price index.
    mode:
      - "start_und_sparplan"
      - "nur_sparplan"
      - "nur_start"
    """
    px = prices.astype(float).dropna()
    if px.empty:
        raise ValueError("Leere Preisreihe in Zukunftspfad.")

    idx = px.index
    shares = 0.0

    # initial buy
    if mode in ("start_und_sparplan", "nur_start") and cfg.start_capital > 0:
        first_px = float(px.iloc[0])
        if first_px > 0:
            shares += cfg.start_capital / first_px

    # contrib dates for future index
    contrib_dates = set()
    if mode in ("start_und_sparplan", "nur_sparplan") and cfg.monthly_contribution > 0:
        start = idx[0]
        end = idx[-1]
        dts = build_contrib_dates(idx, start, end, int(cfg.contrib_day))
        contrib_dates = set(dts)

    nav = []
    for d in idx:
        p = float(px.loc[d])
        if d in contrib_dates and cfg.monthly_contribution > 0 and p > 0:
            buy_px = p * (1.0 + cfg.slippage_bps / 10_000.0)
            shares += cfg.monthly_contribution / buy_px
        nav.append(shares * p)

    return pd.Series(nav, index=idx)


def _invested_amount_future(idx: pd.DatetimeIndex, cfg: BacktestConfig, mode: str) -> float:
    """Deterministic invested amount for MC horizon given the future index."""
    invested = 0.0
    if mode in ("start_und_sparplan", "nur_start"):
        invested += float(cfg.start_capital)
    if mode in ("start_und_sparplan", "nur_sparplan") and cfg.monthly_contribution > 0:
        # count monthly contrib dates in that horizon
        dts = build_contrib_dates(idx, idx[0], idx[-1], int(cfg.contrib_day))
        invested += float(cfg.monthly_contribution) * len(dts)
    return float(invested)


def run_mc_bootstrap(
    base_prices_hist: pd.DataFrame,
    cfg: BacktestConfig,
    mc: MonteCarloConfig,
    leverage: float,
    annual_fin: float,
    annual_fee: float,
) -> MonteCarloResult:
    """
    Bootstrapping Monte Carlo:
    - sample historical daily returns of base
    - build future base price paths
    - build future synthetic leveraged price paths from sampled base returns
    - run DCA path on each (start/sparplan modes)
    Output:
      - summary table (end-value distribution + prob under base)
      - end_values arrays
      - bands (p5/p50/p95) over time
    """
    base_prices_hist = _validate_prices(base_prices_hist)
    hist_px = base_prices_hist["price"].astype(float).dropna()
    hist_rets = hist_px.pct_change().dropna().values
    if hist_rets.size < 50:
        raise ValueError("Zu wenige historische Daten für Monte Carlo (mind. ~50 Renditetage).")

    last_date = hist_px.index[-1]
    n_days = int(mc.horizon_years * cfg.trading_days)
    future_index = pd.bdate_range(start=last_date + timedelta(days=1), periods=n_days)

    base_last = float(hist_px.iloc[-1])

    daily_fin = (1.0 + float(annual_fin)) ** (1.0 / cfg.trading_days) - 1.0
    daily_fee = (1.0 + float(annual_fee)) ** (1.0 / cfg.trading_days) - 1.0

    rng = np.random.default_rng()

    # store portfolio equity paths for bands
    base_paths = np.zeros((n_days, mc.n_paths), dtype="float64")
    syn_paths = np.zeros((n_days, mc.n_paths), dtype="float64")

    base_end = np.zeros(mc.n_paths, dtype="float64")
    syn_end = np.zeros(mc.n_paths, dtype="float64")

    for i in range(mc.n_paths):
        # sample daily returns
        sim_rets = rng.choice(hist_rets, size=n_days, replace=True)

        # base price path
        sim_base_px = base_last * np.cumprod(1.0 + sim_rets)
        s_base = pd.Series(sim_base_px, index=future_index)

        # synthetic leveraged returns from same base returns
        lev_rets = float(leverage) * sim_rets - daily_fin - daily_fee
        lev_rets = np.clip(lev_rets, -0.95, None)
        sim_syn_px = base_last * np.cumprod(1.0 + lev_rets)  # scaled; level not critical for path DCA
        s_syn = pd.Series(sim_syn_px, index=future_index)

        nav_base = _dca_equity_path(s_base, cfg, mc.mode)
        nav_syn = _dca_equity_path(s_syn, cfg, mc.mode)

        base_paths[:, i] = nav_base.values
        syn_paths[:, i] = nav_syn.values
        base_end[i] = float(nav_base.iloc[-1])
        syn_end[i] = float(nav_syn.iloc[-1])

    invested = _invested_amount_future(future_index, cfg, mc.mode)

    def bands(paths: np.ndarray) -> dict:
        return {
            "p5": pd.Series(np.percentile(paths, 5, axis=1), index=future_index),
            "p50": pd.Series(np.percentile(paths, 50, axis=1), index=future_index),
            "p95": pd.Series(np.percentile(paths, 95, axis=1), index=future_index),
        }

    base_band = bands(base_paths)
    syn_band = bands(syn_paths)

    def summary_row(name: str, arr: np.ndarray, ref: np.ndarray | None = None) -> dict:
        prob_under = float(np.mean(arr < ref)) if ref is not None else np.nan
        return {
            "Produkt": name,
            "Investiert (Modus)": invested,
            "Ø Endwert": float(arr.mean()),
            "Median Endwert": float(np.median(arr)),
            "5%-Perzentil": float(np.percentile(arr, 5)),
            "95%-Perzentil": float(np.percentile(arr, 95)),
            "P(Endwert < Basis)": prob_under,
        }

    summary = pd.DataFrame([
        summary_row("Basis-ETF (MC)", base_end, ref=None),
        summary_row(f"Synth {leverage:.1f}x (MC)", syn_end, ref=base_end),
    ])

    return MonteCarloResult(
        summary=summary,
        end_values={"base": base_end, "syn": syn_end},
        bands={"base": base_band, "syn": syn_band},
        index=future_index,
    )
