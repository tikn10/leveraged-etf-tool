import streamlit as st
import pandas as pd
import yfinance as yf

from engine import BacktestConfig, run_dca_backtest, make_synthetic_leverage_prices, TRADING_DAYS_DEFAULT
from metrics import calc_metrics_simple, calc_drawdown, xirr_from_cashflows, make_cashflows_from_contrib_and_final


# -----------------------
# Data loading (cached)
# -----------------------
@st.cache_data(show_spinner=False)
def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Keine Daten für {ticker}. Prüfe Ticker/Zeitraum.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    prices = close.dropna().to_frame("price")
    prices.index = pd.to_datetime(prices.index)
    return prices


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Leveraged ETF Backtesting", layout="wide")
st.title("Leveraged ETF Tool – Backtesting (Basis vs. synthetischer Hebel)")

with st.sidebar:
    st.header("1) Produkt & Zeitraum")
    base_ticker = st.text_input("Basis-ETF / Index Ticker", value="URTH", help="z.B. URTH, ACWI, QQQ, SPY, ^GSPC")
    start = st.date_input("Start", value=pd.to_datetime("2015-01-01")).isoformat()
    end = st.date_input("Ende", value=pd.to_datetime("today")).isoformat()

    st.header("2) Investment (DCA)")
    start_cap = st.number_input("Startkapital", min_value=0.0, value=10_000.0, step=100.0)
    monthly = st.number_input("Monatliche Sparrate", min_value=0.0, value=300.0, step=50.0)
    contrib_day = st.number_input("Spar-Tag (1–28)", min_value=1, max_value=28, value=1, step=1)
    slippage = st.number_input("Slippage pro Kauf (bps)", min_value=0.0, value=0.0, step=1.0)

    st.header("3) Synthetischer Hebel")
    leverage = st.number_input("Hebel (x)", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    annual_fin = st.number_input("Finanzierungskosten p.a. (%)", min_value=0.0, value=2.0, step=0.1) / 100.0
    annual_fee = st.number_input("TER p.a. (%)", min_value=0.0, value=0.23, step=0.01) / 100.0

    st.header("4) Kennzahlen")
    rf = st.number_input("Risikofrei p.a. (%)", min_value=0.0, value=1.0, step=0.1) / 100.0
    tdays = st.number_input("Handelstage/Jahr", min_value=1, value=TRADING_DAYS_DEFAULT, step=1)

    run = st.button("Backtest starten")


# -----------------------
# Run
# -----------------------
if run:
    try:
        with st.spinner("Lade Kursdaten…"):
            base_prices = load_prices(base_ticker.strip(), start, end)

        cfg = BacktestConfig(
            start_capital=float(start_cap),
            monthly_contribution=float(monthly),
            contrib_day=int(contrib_day),
            slippage_bps=float(slippage),
            trading_days=int(tdays),
        )

        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)

        # Backtest basis
        base_res = run_dca_backtest(base_prices, cfg, start_ts, end_ts)

        # Synthetic leveraged series + backtest
        syn_prices = make_synthetic_leverage_prices(
            base_prices=base_prices, leverage=float(leverage),
            annual_fin=float(annual_fin), annual_fee=float(annual_fee),
            trading_days=int(tdays),
        )
        syn_res = run_dca_backtest(syn_prices, cfg, start_ts, end_ts)

        # Headline metrics
        c1, c2 = st.columns(2)
        c1.subheader("Basis")
        c1.metric("Endwert", f"{base_res.summary['final_value']:,.2f}")
        c1.metric("P/L %", f"{base_res.summary['pl_pct']:,.2f}%")

        c2.subheader(f"Synth {leverage:.1f}x")
        c2.metric("Endwert", f"{syn_res.summary['final_value']:,.2f}")
        c2.metric("P/L %", f"{syn_res.summary['pl_pct']:,.2f}%")

        # KPI table
        m_base = calc_metrics_simple(base_res.equity, rf_annual=rf, trading_days=int(tdays))
        m_syn = calc_metrics_simple(syn_res.equity, rf_annual=rf, trading_days=int(tdays))

        kpi = pd.DataFrame([m_base, m_syn], index=[f"Basis {base_ticker}", f"Synth {leverage:.1f}x"])
        st.subheader("Kennzahlen (einfach & stabil)")
        st.dataframe(
            kpi.style.format({"CAGR": "{:.2%}", "Vol": "{:.2%}", "Sharpe": "{:.2f}", "MaxDD": "{:.2%}"})
        )

        # XIRR (DCA korrekt)
        cf_base = make_cashflows_from_contrib_and_final(base_res.contrib, base_res.equity.iloc[-1])
        cf_syn = make_cashflows_from_contrib_and_final(syn_res.contrib, syn_res.equity.iloc[-1])
        irr_base = xirr_from_cashflows(cf_base)
        irr_syn = xirr_from_cashflows(cf_syn)

        st.caption(f"XIRR (DCA): Basis {irr_base:.2%} | Synth {leverage:.1f}x {irr_syn:.2%}")

        # Charts
        st.subheader("Equity + Einzahlungen")
        chart_df = pd.DataFrame(
            {
                f"Basis {base_ticker}": base_res.equity,
                f"Synth {leverage:.1f}x": syn_res.equity,
                "Einzahlungen (kum.)": base_res.contrib,
            }
        )
        st.line_chart(chart_df)

        st.subheader("Drawdowns")
        dd_df = pd.DataFrame(
            {
                f"Basis {base_ticker}": calc_drawdown(base_res.equity),
                f"Synth {leverage:.1f}x": calc_drawdown(syn_res.equity),
            }
        )
        st.line_chart(dd_df)

        # Downloads
        st.subheader("Downloads")
        base_out = pd.DataFrame({"equity": base_res.equity, "contrib": base_res.contrib, "shares": base_res.shares})
        syn_out = pd.DataFrame({"equity": syn_res.equity, "contrib": syn_res.contrib, "shares": syn_res.shares})

        st.download_button(
            "CSV Basis",
            data=base_out.to_csv(index=True).encode("utf-8"),
            file_name=f"backtest_basis_{base_ticker}.csv",
            mime="text/csv",
        )
        st.download_button(
            "CSV Synth",
            data=syn_out.to_csv(index=True).encode("utf-8"),
            file_name=f"backtest_synth_{leverage:.1f}x.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(str(e))
else:
    st.info("Links die Parameter wählen und **Backtest starten** klicken.")
