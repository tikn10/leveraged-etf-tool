from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import numpy as np
import pandas as pd

TRADING_DAYS_DEFAULT = 252


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
    ci_level: float = 0.95  # 0.90 oder 0.95
    seed: int | None = None


@dataclass
class MonteCarloResult:
    summary: pd.DataFrame
    end_values: dict          # {"base": np.ndarray, "syn": np.ndarray}
    bands: dict               # {"base": {"p_low": Series, "p50": Series, "p_high": Series}, "syn": {...}}
    index: pd.DatetimeIndex   # future index

    assumptions: dict         # Annahmen/Herleitungen für UI
    path_metrics: dict        # Arrays pro Pfad (für Downloads / weitere Auswertungen)
    risk_probs: dict          # Wahrscheinlichkeiten (Loss, Underperf, MaxDD-Schwellen, etc.)


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
    prices = _validate_prices(prices)
    px = prices.loc[start:end, "price"].astype(float).dropna()
    if px.empty:
        raise ValueError("Keine Preisdaten im gewählten Zeitraum.")

    idx = px.index
    contrib_dates = build_contrib_dates(idx, start, end, int(cfg.contrib_day))

    shares = pd.Series(index=idx, dtype="float64")

    first_px = float(px.iloc[0])
    shares.iloc[0] = (cfg.start_capital / first_px) if first_px > 0 else 0.0

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


def _dca_equity_path(prices: pd.Series, cfg: BacktestConfig, mode: str) -> pd.Series:
    px = prices.astype(float).dropna()
    if px.empty:
        raise ValueError("Leere Preisreihe in Zukunftspfad.")

    idx = px.index
    shares = 0.0

    if mode in ("start_und_sparplan", "nur_start") and cfg.start_capital > 0:
        first_px = float(px.iloc[0])
        if first_px > 0:
            shares += cfg.start_capital / first_px

    contrib_dates = set()
    if mode in ("start_und_sparplan", "nur_sparplan") and cfg.monthly_contribution > 0:
        dts = build_contrib_dates(idx, idx[0], idx[-1], int(cfg.contrib_day))
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
    invested = 0.0
    if mode in ("start_und_sparplan", "nur_start"):
        invested += float(cfg.start_capital)
    if mode in ("start_und_sparplan", "nur_sparplan") and cfg.monthly_contribution > 0:
        dts = build_contrib_dates(idx, idx[0], idx[-1], int(cfg.contrib_day))
        invested += float(cfg.monthly_contribution) * len(dts)
    return float(invested)


def _max_drawdown_and_ttr_days_from_path(path_values: np.ndarray) -> tuple[float, float | None]:
    """
    path_values: 1D array of portfolio value over time.
    - max_drawdown: negative number (e.g. -0.55)
    - ttr_days: days from peak before maxDD to first recovery back to that peak value.
    """
    if path_values.size < 3:
        return float("nan"), None

    peak = np.maximum.accumulate(path_values)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = path_values / np.maximum(peak, 1e-12) - 1.0

    trough_i = int(np.nanargmin(dd))
    maxdd = float(dd[trough_i])

    # peak index (first occurrence of peak value up to trough)
    peak_val = float(np.nanmax(peak[: trough_i + 1]))
    # earliest index of that peak value
    peak_i = int(np.argmax(peak[: trough_i + 1] >= peak_val - 1e-12))

    after = path_values[trough_i:]
    rec_idx = np.where(after >= peak_val)[0]
    if rec_idx.size == 0:
        return maxdd, None

    recovery_i = trough_i + int(rec_idx[0])
    # Business-day index -> "days" approximieren als Trading Days
    ttr_days = float(recovery_i - peak_i)
    return maxdd, ttr_days


def run_mc_bootstrap(
    base_prices_hist: pd.DataFrame,
    cfg: BacktestConfig,
    mc: MonteCarloConfig,
    leverage: float,
    annual_fin: float,
    annual_fee: float,
    rf_annual: float = 0.0,
) -> MonteCarloResult:
    base_prices_hist = _validate_prices(base_prices_hist)
    hist_px = base_prices_hist["price"].astype(float).dropna()
    hist_rets = hist_px.pct_change().dropna().values
    if hist_rets.size < 50:
        raise ValueError("Zu wenige historische Daten für Monte Carlo (mind. ~50 Renditetage).")

    ci = float(mc.ci_level)
    if ci not in (0.90, 0.95):
        raise ValueError("ci_level muss 0.90 oder 0.95 sein.")

    alpha = (1.0 - ci) / 2.0
    p_low = 100.0 * alpha
    p_high = 100.0 * (1.0 - alpha)

    # Historische Annahmen (aus Stichprobe)
    hist_start = hist_px.index[0]
    hist_end = hist_px.index[-1]
    hist_days = int(hist_rets.size)
    mu_d = float(np.mean(hist_rets))
    sig_d = float(np.std(hist_rets, ddof=1))
    ann_ret_hist = (1.0 + mu_d) ** cfg.trading_days - 1.0
    ann_vol_hist = sig_d * np.sqrt(cfg.trading_days)

    daily_fin = (1.0 + float(annual_fin)) ** (1.0 / cfg.trading_days) - 1.0
    daily_fee = (1.0 + float(annual_fee)) ** (1.0 / cfg.trading_days) - 1.0

    # Erwartungswerte für synthetisch (auf Basis der historischen Tagesrenditen, nicht "garantiert")
    mu_d_syn = float(leverage) * mu_d - daily_fin - daily_fee
    sig_d_syn = abs(float(leverage)) * sig_d
    ann_ret_syn = (1.0 + mu_d_syn) ** cfg.trading_days - 1.0
    ann_vol_syn = sig_d_syn * np.sqrt(cfg.trading_days)

    last_date = hist_px.index[-1]
    n_days = int(mc.horizon_years * cfg.trading_days)
    future_index = pd.bdate_range(start=last_date + timedelta(days=1), periods=n_days)

    base_last = float(hist_px.iloc[-1])

    rng = np.random.default_rng(mc.seed)

    base_paths = np.zeros((n_days, mc.n_paths), dtype="float64")
    syn_paths = np.zeros((n_days, mc.n_paths), dtype="float64")

    base_end = np.zeros(mc.n_paths, dtype="float64")
    syn_end = np.zeros(mc.n_paths, dtype="float64")

    # per-path metrics
    base_maxdd = np.full(mc.n_paths, np.nan, dtype="float64")
    syn_maxdd = np.full(mc.n_paths, np.nan, dtype="float64")
    base_ttr = np.full(mc.n_paths, np.nan, dtype="float64")  # trading days
    syn_ttr = np.full(mc.n_paths, np.nan, dtype="float64")

    base_sharpe = np.full(mc.n_paths, np.nan, dtype="float64")
    syn_sharpe = np.full(mc.n_paths, np.nan, dtype="float64")

    rf_daily = (1.0 + float(rf_annual)) ** (1.0 / cfg.trading_days) - 1.0

    for i in range(mc.n_paths):
        sim_rets = rng.choice(hist_rets, size=n_days, replace=True)

        sim_base_px = base_last * np.cumprod(1.0 + sim_rets)
        s_base = pd.Series(sim_base_px, index=future_index)

        lev_rets = float(leverage) * sim_rets - daily_fin - daily_fee
        lev_rets = np.clip(lev_rets, -0.95, None)
        sim_syn_px = base_last * np.cumprod(1.0 + lev_rets)
        s_syn = pd.Series(sim_syn_px, index=future_index)

        nav_base = _dca_equity_path(s_base, cfg, mc.mode)
        nav_syn = _dca_equity_path(s_syn, cfg, mc.mode)

        base_paths[:, i] = nav_base.values
        syn_paths[:, i] = nav_syn.values
        base_end[i] = float(nav_base.iloc[-1])
        syn_end[i] = float(nav_syn.iloc[-1])

        # MaxDD + TTR (auf Portfolio-Wertpfad)
        mdd_b, ttr_b = _max_drawdown_and_ttr_days_from_path(base_paths[:, i])
        mdd_s, ttr_s = _max_drawdown_and_ttr_days_from_path(syn_paths[:, i])
        base_maxdd[i] = mdd_b
        syn_maxdd[i] = mdd_s
        if ttr_b is not None:
            base_ttr[i] = float(ttr_b)
        if ttr_s is not None:
            syn_ttr[i] = float(ttr_s)

        # Sharpe auf *Underlying*-Tagesrenditen (nicht Equity inkl. Einzahlungen)
        # => für Interpretation sauberer
        sd = float(np.std(sim_rets, ddof=1))
        if sd > 0:
            base_sharpe[i] = float(((np.mean(sim_rets - rf_daily)) / sd) * np.sqrt(cfg.trading_days))

        sd2 = float(np.std(lev_rets, ddof=1))
        if sd2 > 0:
            syn_sharpe[i] = float(((np.mean(lev_rets - rf_daily)) / sd2) * np.sqrt(cfg.trading_days))

    invested = _invested_amount_future(future_index, cfg, mc.mode)

    def _bands(paths: np.ndarray) -> dict:
        return {
            "p_low": pd.Series(np.percentile(paths, p_low, axis=1), index=future_index),
            "p50": pd.Series(np.percentile(paths, 50, axis=1), index=future_index),
            "p_high": pd.Series(np.percentile(paths, p_high, axis=1), index=future_index),
        }

    base_band = _bands(base_paths)
    syn_band = _bands(syn_paths)

    # Total Return und "CAGR" (vereinfachte Definition über Summe Einzahlungen)
    total_return_base = base_end / max(invested, 1e-12) - 1.0
    total_return_syn = syn_end / max(invested, 1e-12) - 1.0

    cagr_base = (base_end / max(invested, 1e-12)) ** (1.0 / max(mc.horizon_years, 1e-12)) - 1.0
    cagr_syn = (syn_end / max(invested, 1e-12)) ** (1.0 / max(mc.horizon_years, 1e-12)) - 1.0

    # VaR/ES (auf Total Return)
    var_cut = np.percentile(total_return_base, 100.0 * (1.0 - ci))  # z.B. 5% links bei 95%
    # (sauberer: alpha=0.05/0.10 – hier direkt aus CI abgeleitet)
    q_alpha = np.percentile(total_return_base, 100.0 * (1.0 - ci))
    es_base = float(np.mean(total_return_base[total_return_base <= q_alpha]))
    q_alpha_s = np.percentile(total_return_syn, 100.0 * (1.0 - ci))
    es_syn = float(np.mean(total_return_syn[total_return_syn <= q_alpha_s]))

    def _ci(arr: np.ndarray) -> tuple[float, float]:
        return (
            float(np.percentile(arr, p_low)),
            float(np.percentile(arr, p_high)),
        )
    
    ci_label = f"{p_low:g}–{p_high:g}% CI"   # z.B. "2.5–97.5% CI" oder "5–95% CI"

    def summary_row(name: str, end_arr: np.ndarray, cagr_arr: np.ndarray, tr_arr: np.ndarray, ref_end: np.ndarray | None = None) -> dict:
        prob_under = float(np.mean(end_arr < ref_end)) if ref_end is not None else np.nan
        end_ci = _ci(end_arr)
        cagr_ci = _ci(cagr_arr)
        tr_ci = _ci(tr_arr)
        return {
            "Produkt": name,
            "Investiert (Modus)": invested,

            "Median Endwert": float(np.median(end_arr)),
            f"{ci_label} Endwert": _ci(end_arr),

            "Median CAGR": float(np.median(cagr_arr)),
            f"{ci_label} CAGR": _ci(cagr_arr),

            "Median Gesamtperformance": float(np.median(tr_arr)),
            f"{ci_label} Gesamtperformance": _ci(tr_arr),

            "P(Endwert < Basis)": prob_under,
        }


    summary = pd.DataFrame([
        summary_row("Basis-ETF (MC)", base_end, cagr_base, total_return_base, ref_end=None),
        summary_row(f"Synth {leverage:.1f}x (MC)", syn_end, cagr_syn, total_return_syn, ref_end=base_end),
    ])

    # Risk probabilities
    maxdd_thresholds = [-0.10, -0.20, -0.30, -0.40, -0.50, -0.60, -0.70, -0.80, -0.90, -1.00]
    probs_base_mdd = {thr: float(np.mean(base_maxdd <= thr)) for thr in maxdd_thresholds}
    probs_syn_mdd = {thr: float(np.mean(syn_maxdd <= thr)) for thr in maxdd_thresholds}

    risk_probs = {
        "loss_prob_base": float(np.mean(base_end < invested)),
        "loss_prob_syn": float(np.mean(syn_end < invested)),
        "underperf_prob_syn": float(np.mean(syn_end < base_end)),  # LETF < 1x
        "maxdd_probs_base": probs_base_mdd,
        "maxdd_probs_syn": probs_syn_mdd,
        "var_base": float(q_alpha),
        "es_base": float(es_base),
        "var_syn": float(q_alpha_s),
        "es_syn": float(es_syn),
    }

    # Assumptions

    ann_ret_base = (1.0 + mu_d) ** cfg.trading_days - 1.0
    ann_vol_base = sig_d * np.sqrt(cfg.trading_days)

    ann_ret_syn = (1.0 + mu_d_syn) ** cfg.trading_days - 1.0
    ann_vol_syn = sig_d_syn * np.sqrt(cfg.trading_days)

    assumptions = {
        "historische_daten": {
            "Zeitraum": f"{hist_px.index[0].date()} bis {hist_px.index[-1].date()}",
            "Tage": int(len(hist_rets)),
        },
        "basis_etf": {
            "Rendite p.a. (implizit)": ann_ret_base,
            "Volatilität p.a.": ann_vol_base,
        },
        "synth_etf": {
            "Rendite p.a. (implizit)": ann_ret_syn,
            "Volatilität p.a.": ann_vol_syn,
        },
    }


    path_metrics = {
        "end_base": base_end,
        "end_syn": syn_end,
        "total_return_base": total_return_base,
        "total_return_syn": total_return_syn,
        "cagr_base": cagr_base,
        "cagr_syn": cagr_syn,
        "maxdd_base": base_maxdd,
        "maxdd_syn": syn_maxdd,
        "ttr_days_base": base_ttr,  # trading days
        "ttr_days_syn": syn_ttr,
        "sharpe_base": base_sharpe,
        "sharpe_syn": syn_sharpe,
        "invested": invested,
    }

    return MonteCarloResult(
        summary=summary,
        end_values={"base": base_end, "syn": syn_end},
        bands={"base": base_band, "syn": syn_band},
        index=future_index,
        assumptions=assumptions,
        path_metrics=path_metrics,
        risk_probs=risk_probs,
    )
