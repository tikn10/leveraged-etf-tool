from __future__ import annotations

import numpy as np
import pandas as pd


def calc_drawdown(equity: pd.Series) -> pd.Series:
    eq = equity.astype(float).dropna()
    if eq.empty:
        return pd.Series(dtype="float64")
    peak = eq.cummax()
    return eq / peak - 1.0


def time_to_recovery(equity: pd.Series) -> float | None:
    """
    Time to Recovery (TTR):
    Zeit vom Beginn des maximalen Drawdowns bis zum Wiedererreichen des vorherigen Allzeithochs.
    Rückgabe in Tagen. Falls kein neues Hoch erreicht wird: None.
    """
    eq = equity.astype(float).dropna()
    if len(eq) < 3:
        return None

    peak = eq.cummax()
    dd = eq / peak - 1.0
    if dd.empty:
        return None

    trough_date = dd.idxmin()
    if pd.isna(trough_date):
        return None

    # Peak-Zeitpunkt direkt vor/bei dem Trough (Beginn des MaxDD)
    peak_to_trough = peak.loc[:trough_date]
    if peak_to_trough.empty:
        return None
    peak_value = float(peak_to_trough.max())
    peak_date = peak_to_trough.idxmax()

    # Recovery: erstes Datum nach dem Trough, an dem Equity wieder >= Peak vor dem Drawdown ist
    after = eq.loc[trough_date:]
    rec = after[after >= peak_value]
    if rec.empty:
        return None

    recovery_date = rec.index[0]
    return float((recovery_date - peak_date).days)


def _window_size_days(window_years: int, trading_days: int) -> int:
    if window_years <= 0 or trading_days <= 0:
        return 0
    return int(window_years * trading_days)


def rolling_loss_share(equity: pd.Series, window_years: int, trading_days: int = 252) -> float:
    """
    Anteil der Rolling-Fenster, in denen Endwert < Startwert.
    Fensterlänge = window_years * trading_days.
    Rückgabe als Anteil [0..1]. Falls zu wenig Daten: NaN.
    """
    eq = equity.astype(float).dropna()
    w = _window_size_days(window_years, trading_days)
    if len(eq) < (w + 1) or w <= 0:
        return float("nan")

    start = eq.iloc[:-w].values
    end = eq.iloc[w:].values
    if len(start) == 0:
        return float("nan")

    share = float(np.mean(end < start))
    return share

def rolling_loss_stats(
    equity: pd.Series,
    window_years: int,
    trading_days: int = 252
) -> tuple[float, int]:
    """
    Wie rolling_loss_share, aber liefert zusätzlich die Anzahl der Fenster.
    Returns:
      - share: Anteil [0..1]
      - n_windows: Anzahl der Rolling-Fenster
    """
    eq = equity.astype(float).dropna()
    w = _window_size_days(window_years, trading_days)
    if len(eq) < (w + 1) or w <= 0:
        return float("nan"), 0

    start = eq.iloc[:-w].values
    end = eq.iloc[w:].values
    n_windows = len(start)
    if n_windows == 0:
        return float("nan"), 0

    share = float(np.mean(end < start))
    return share, int(n_windows)


def rolling_underperformance_share(
    lev_equity: pd.Series,
    base_equity: pd.Series,
    window_years: int,
    trading_days: int = 252,
) -> float:
    """
    Anteil der Rolling-Fenster, in denen LETF (lev) schlechter performt als 1x (base):
      (lev_end / lev_start) < (base_end / base_start)
    Fensterlänge = window_years * trading_days.
    Rückgabe als Anteil [0..1]. Falls zu wenig Daten: NaN.
    """
    lev = lev_equity.astype(float).dropna()
    base = base_equity.astype(float).dropna()

    df = pd.concat([lev.rename("lev"), base.rename("base")], axis=1).dropna()
    w = _window_size_days(window_years, trading_days)
    if len(df) < (w + 1) or w <= 0:
        return float("nan")

    lev_start = df["lev"].iloc[:-w].values
    lev_end = df["lev"].iloc[w:].values
    base_start = df["base"].iloc[:-w].values
    base_end = df["base"].iloc[w:].values

    # Schutz gegen Division durch 0 (sollte praktisch nicht vorkommen, aber robust)
    eps = 1e-12
    lev_start = np.maximum(lev_start, eps)
    base_start = np.maximum(base_start, eps)

    lev_ret = lev_end / lev_start
    base_ret = base_end / base_start

    share = float(np.mean(lev_ret < base_ret))
    return share

def rolling_underperformance_stats(
    lev_equity: pd.Series,
    base_equity: pd.Series,
    window_years: int,
    trading_days: int = 252
) -> tuple[float, int]:
    """
    Wie rolling_underperformance_share, aber liefert zusätzlich die Anzahl der Fenster.
    Returns:
      - share: Anteil [0..1]
      - n_windows: Anzahl der Rolling-Fenster
    """
    lev = lev_equity.astype(float).dropna()
    base = base_equity.astype(float).dropna()

    df = pd.concat([lev.rename("lev"), base.rename("base")], axis=1).dropna()
    w = _window_size_days(window_years, trading_days)
    if len(df) < (w + 1) or w <= 0:
        return float("nan"), 0

    lev_start = df["lev"].iloc[:-w].values
    lev_end = df["lev"].iloc[w:].values
    base_start = df["base"].iloc[:-w].values
    base_end = df["base"].iloc[w:].values

    n_windows = len(lev_start)
    if n_windows == 0:
        return float("nan"), 0

    eps = 1e-12
    lev_start = np.maximum(lev_start, eps)
    base_start = np.maximum(base_start, eps)

    lev_ret = lev_end / lev_start
    base_ret = base_end / base_start

    share = float(np.mean(lev_ret < base_ret))
    return share, int(n_windows)

def rolling_severe_loss_share(
    equity: pd.Series,
    window_years: int,
    loss_threshold: float = -0.2,
    trading_days: int = 252
) -> float:
    """
    Anteil der Rolling-Fenster, in denen (Endwert/Startwert - 1) < loss_threshold.
    loss_threshold z.B. -0.2 für -20%.
    """
    eq = equity.astype(float).dropna()
    w = _window_size_days(window_years, trading_days)
    if len(eq) < (w + 1) or w <= 0:
        return float("nan")

    start = eq.iloc[:-w].values
    end = eq.iloc[w:].values
    if len(start) == 0:
        return float("nan")

    ret = (end / np.maximum(start, 1e-12)) - 1.0
    return float(np.mean(ret < loss_threshold))


def calc_metrics_simple(equity: pd.Series, rf_annual: float = 0.0, trading_days: int = 252) -> dict:
    eq = equity.astype(float).dropna()
    if len(eq) < 3:
        return {}

    rets = eq.pct_change().dropna()
    if rets.empty:
        return {}

    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    vol = float(rets.std() * np.sqrt(trading_days))

    rf_daily = (1.0 + float(rf_annual)) ** (1.0 / trading_days) - 1.0
    sharpe = np.nan
    if rets.std() > 0:
        sharpe = float(((rets - rf_daily).mean() / rets.std()) * np.sqrt(trading_days))

    dd = calc_drawdown(eq)
    maxdd = float(dd.min()) if not dd.empty else np.nan

    # Calmar = CAGR / |MaxDD|
    calmar = np.nan
    if maxdd is not None and not np.isnan(maxdd) and maxdd < 0:
        if cagr is not None and not np.isnan(cagr):
            calmar = float(cagr / abs(maxdd))

    return {"CAGR": float(cagr), "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd, "Calmar": calmar}


def xirr_from_cashflows(cashflows: pd.Series) -> float:
    cf = cashflows.dropna()
    if cf.empty:
        return np.nan

    t0 = cf.index[0]
    years = (cf.index - t0).days / 365.25
    amounts = cf.values.astype(float)

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** years))

    def d_npv(rate: float) -> float:
        return float(np.sum(-years * amounts / (1.0 + rate) ** (years + 1.0)))

    r = 0.07
    for _ in range(60):
        f = npv(r)
        df = d_npv(r)
        if abs(df) < 1e-12:
            break
        step = f / df
        r -= step
        if abs(step) < 1e-10:
            break

    return float(r)


def make_cashflows_from_contrib_and_final(contrib_cum: pd.Series, final_value: float) -> pd.Series:
    c = contrib_cum.astype(float).dropna()
    if c.empty:
        return pd.Series(dtype="float64")

    deposits = c.diff().fillna(c.iloc[0])
    cf = -deposits
    cf.iloc[-1] = cf.iloc[-1] + float(final_value)
    return cf

def var_es(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """
    Value at Risk & Expected Shortfall auf Verteilung x (z.B. Total Return).
    alpha=0.05 => 5%-Quantil (linke Seite).
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (np.nan, np.nan)

    var = float(np.quantile(arr, alpha))
    tail = arr[arr <= var]
    es = float(np.mean(tail)) if tail.size else np.nan
    return var, es
