from __future__ import annotations

import numpy as np
import pandas as pd


def calc_drawdown(equity: pd.Series) -> pd.Series:
    eq = equity.astype(float).dropna()
    if eq.empty:
        return pd.Series(dtype="float64")
    peak = eq.cummax()
    return eq / peak - 1.0


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

    return {"CAGR": float(cagr), "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd}


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
