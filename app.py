import streamlit as st
import pandas as pd
import yfinance as yf

from engine import (
    BacktestConfig,
    MonteCarloConfig,
    run_dca_backtest,
    make_synthetic_leverage_prices,
    run_mc_bootstrap,
    TRADING_DAYS_DEFAULT,
)
from metrics import calc_metrics_simple, calc_drawdown, xirr_from_cashflows, make_cashflows_from_contrib_and_final

from datetime import date

MIN_DATE = pd.to_datetime("1950-01-01")
MAX_DATE = pd.to_datetime("today")

@st.cache_data(show_spinner=False)
def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"Keine Daten für {ticker}. Prüfe Ticker/Zeitraum.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    prices = close.dropna().to_frame("price")
    prices.index = pd.to_datetime(prices.index)
    return prices

@st.cache_data(show_spinner=False)
def get_ticker_meta(ticker: str):
    """
    Returns (display_name, first_date_available)
    """
    t = (ticker or "").strip()
    if not t:
        return ("", None)

    try:
        yt = yf.Ticker(t)

        # Name
        try:
            info = yt.info or {}
            name = info.get("longName") or info.get("shortName") or t
        except Exception:
            name = t

        # Historie (maximal verfügbar)
        hist = yt.history(period="max", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return (name, None)

        # Zeitzone entfernen
        first_dt = pd.to_datetime(hist.index.min()).tz_localize(None).normalize()

        return (name, first_dt)

    except Exception:
        return (t, None)


st.set_page_config(page_title="Leveraged ETF Tool", layout="wide")
st.title("Leveraged ETF Tool – Backtesting & Monte Carlo")


# -------------------------------------------------
# Gemeinsamer Block: Basis-ETF + Zeitraum
# -------------------------------------------------
st.subheader("Grundeinstellungen") 

st.divider()

st.markdown("#### Basis-ETF")


c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    base_ticker = st.text_input(
        "Basis-ETF / Index Ticker",
        value="URTH",
        help="z.B. URTH, ACWI, QQQ, SPY, ^GSPC"
    ).strip()

display_name, first_date = get_ticker_meta(base_ticker) if base_ticker else ("", None)

# Anzeige: voller Name + frühestes Datum
if base_ticker:
    if first_date is not None:
        st.caption(
            f"Auswahl: **{display_name}** (`{base_ticker}`) · "
            f"Frühestes verfügbares Datum: **{first_date.date()}**"
        )
    else:
        st.caption(f"Auswahl: **{display_name}** (`{base_ticker}`) · Frühestes Datum: unbekannt (keine Historie gefunden)")

# Für DatePicker: min_value = max(1950, first_date) (falls first_date vorhanden)
min_start = MIN_DATE
if first_date is not None:
    first_date = pd.to_datetime(first_date).tz_localize(None)
    min_start = max(MIN_DATE, first_date)

with c2:
    start_dt = st.date_input(
        "Start Historie",
        value=pd.to_datetime("2015-01-01"),
        min_value=min_start.to_pydatetime().date(),
        max_value=MAX_DATE.to_pydatetime().date(),
        key="start_date",
        help="Startdatum der verfügbaren historischen Kursdaten (ab 1950)"
    )
with c3:
    end_dt = st.date_input(
        "Ende Historie",
        value=MAX_DATE.to_pydatetime().date(),
        min_value=min_start.to_pydatetime().date(),
        max_value=MAX_DATE.to_pydatetime().date(),
        key="end_date",
    )

start = pd.to_datetime(start_dt).date().isoformat()  # "YYYY-MM-DD"
end = pd.to_datetime(end_dt).date().isoformat()      # "YYYY-MM-DD"

st.markdown("#### Investment")

i1, i2, i3, i4 = st.columns([1, 1, 1, 1])
with i1:
    common_start_cap = st.number_input(
        "Startkapital",
        min_value=0.0,
        value=10_000.0,
        step=100.0,
        key="common_start",
        help="Nur Sparrate? Dann Startkapital = 0. Nur Startkapital (Buy&Hold)? Dann Sparrate = 0."
    )
with i2:
    common_monthly = st.number_input(
        "Monatliche Sparrate",
        min_value=0.0,
        value=300.0,
        step=50.0,
        key="common_monthly",
        help="Nur Startkapital (Buy&Hold)? Sparrate = 0. Nur Sparrate? Startkapital = 0."
    )

with i3:
    common_day = st.number_input("Spar-Tag (1–28)", min_value=1, max_value=28, value=1, step=1, key="common_day")
with i4:
    common_slippage = st.number_input("Slippage pro Kauf (bps)", min_value=0.0, value=0.0, step=1.0, key="common_slip")

st.markdown("#### Synthetischer Hebel")


d1, d2, d3, d4 = st.columns([1, 1, 1, 1])
# Hebel

with d1:
        leverage = st.number_input(
            "Hebel (x)",
            min_value=1.0,
            max_value=100.0,
            value=2.0,
            step=0.5,
            key="bt_lev"
        )

with d2:
        annual_fin = st.number_input(
            "Finanzierungskosten p.a. (%)",
            min_value=0.0,
            value=2.0,
            step=0.1,
            key="bt_fin",
            help="Wird täglich auf die synthetische Reihe umgelegt."
        ) / 100.0

with d3:
        annual_fee = st.number_input(
            "TER p.a. (%)",
            min_value=0.0,
            value=0.23,
            step=0.01,
            key="bt_fee",
            help="Wird täglich auf die synthetische Reihe umgelegt."
        ) / 100.0

with d4:
        tdays = st.number_input(
        "Handelstage/Jahr",
        min_value=1,
        value=TRADING_DAYS_DEFAULT,
        step=1,
        key="bt_tdays",
        help="Standard: 252"
    )    


st.divider()
tabs = st.tabs(["📈 Backtesting", "🔮 Monte Carlo"])


# -------------------------------------------------
# Tab 1: Backtesting
# -------------------------------------------------
with tabs[0]:
    st.subheader("Backtesting")

    st.markdown("#### Einstellungen")

    e1, _ = st.columns([1, 3])
    
    with e1:
        rf = st.number_input(
            "Risikofrei p.a. (%)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key="bt_rf",
            help="Wird für Sharpe genutzt (annualisiert)."
        ) / 100.0
    
    run_bt = st.button("Backtesting starten", key="run_bt")

    if run_bt:
        try:
            with st.spinner("Lade Kursdaten…"):
                base_prices = load_prices(base_ticker.strip(), start, end)

            cfg = BacktestConfig(
                start_capital=float(common_start_cap),
                monthly_contribution=float(common_monthly),
                contrib_day=int(common_day),
                slippage_bps=float(common_slippage),
                trading_days=int(tdays),
            )

            start_ts = pd.to_datetime(start)
            end_ts = pd.to_datetime(end)

            base_res = run_dca_backtest(base_prices, cfg, start_ts, end_ts)

            syn_prices = make_synthetic_leverage_prices(
                base_prices=base_prices,
                leverage=float(leverage),
                annual_fin=float(annual_fin),
                annual_fee=float(annual_fee),
                trading_days=int(tdays),
            )
            syn_res = run_dca_backtest(syn_prices, cfg, start_ts, end_ts)

            c1, c2 = st.columns(2)
            c1.metric("Basis Endwert", f"{base_res.summary['final_value']:,.2f}")
            c1.metric("Basis P/L %", f"{base_res.summary['pl_pct']:,.2f}%")
            c2.metric(f"Synth {leverage:.1f}x Endwert", f"{syn_res.summary['final_value']:,.2f}")
            c2.metric("Synth P/L %", f"{syn_res.summary['pl_pct']:,.2f}%")

            m_base = calc_metrics_simple(base_res.equity, rf_annual=rf, trading_days=int(tdays))
            m_syn = calc_metrics_simple(syn_res.equity, rf_annual=rf, trading_days=int(tdays))
            kpi = pd.DataFrame([m_base, m_syn], index=[f"Basis {base_ticker}", f"Synth {leverage:.1f}x"])
            st.subheader("Kennzahlen")
            st.dataframe(
                kpi.style.format({"CAGR": "{:.2%}", "Vol": "{:.2%}", "Sharpe": "{:.2f}", "MaxDD": "{:.2%}"})
            )

            cf_base = make_cashflows_from_contrib_and_final(base_res.contrib, base_res.equity.iloc[-1])
            cf_syn = make_cashflows_from_contrib_and_final(syn_res.contrib, syn_res.equity.iloc[-1])
            irr_base = xirr_from_cashflows(cf_base)
            irr_syn = xirr_from_cashflows(cf_syn)
            st.caption(f"XIRR (DCA): Basis {irr_base:.2%} | Synth {leverage:.1f}x {irr_syn:.2%}")

            st.subheader("Equity + Einzahlungen")
            st.line_chart(pd.DataFrame({
                f"Basis {base_ticker}": base_res.equity,
                f"Synth {leverage:.1f}x": syn_res.equity,
                "Einzahlungen (kum.)": base_res.contrib,
            }))

            st.subheader("Drawdowns")
            st.line_chart(pd.DataFrame({
                f"Basis {base_ticker}": calc_drawdown(base_res.equity),
                f"Synth {leverage:.1f}x": calc_drawdown(syn_res.equity),
            }))

            st.subheader("Downloads")
            base_out = pd.DataFrame({"equity": base_res.equity, "contrib": base_res.contrib, "shares": base_res.shares})
            syn_out = pd.DataFrame({"equity": syn_res.equity, "contrib": syn_res.contrib, "shares": syn_res.shares})

            st.download_button(
                "CSV Basis",
                base_out.to_csv(index=True).encode("utf-8"),
                file_name=f"backtest_basis_{base_ticker}.csv",
                mime="text/csv"
            )
            st.download_button(
                "CSV Synth",
                syn_out.to_csv(index=True).encode("utf-8"),
                file_name=f"backtest_synth_{leverage:.1f}x.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(str(e))

# -------------------------------------------------
# Tab 2: Monte Carlo
# -------------------------------------------------
with tabs[1]:
    st.subheader("Monte Carlo-Simulation")

    st.markdown("#### Einstellungen")
    
    f1, f2, f3, _ = st.columns([1, 1, 1, 1])

    with f1:
        mc_horizon = st.number_input(
            "Horizont (Jahre)",
            min_value=1,
            max_value=40,
            value=10,
            step=1,
            key="mc_h",
            help="Wie viele Jahre ab heute simuliert werden."
        )
    with f2:
        mc_paths = st.number_input(
            "Anzahl Pfade",
            min_value=200,
            max_value=10000,
            value=1000,
            step=100,
            key="mc_n",
            help="Mehr Pfade = stabilere Ergebnisse, aber langsamer."
        )
    with f3:
        mc_mode_label = st.selectbox(
            "Investitionsmodus",
            ["Startkapital + Sparrate", "Nur Sparrate", "Nur Startkapital (Buy&Hold)"],
            key="mc_mode",
            help="Steuert, ob im Zukunftszeitraum Startkapital und/oder Sparrate investiert wird."
        )

    # Übersetze Label -> Mode (für engine)
    if mc_mode_label == "Startkapital + Sparrate":
        mc_mode = "start_und_sparplan"
    elif mc_mode_label == "Nur Sparrate":
        mc_mode = "nur_sparplan"
    else:
        mc_mode = "nur_start"

    # Optional: kleine Plausibilitäts-Warnung (hilft Laien)
    if mc_mode == "nur_start" and common_monthly > 0:
        st.warning("Du hast 'Nur Startkapital' gewählt, aber eine Sparrate > 0 gesetzt. "
                "Das ist ok, aber die Sparrate wird im MC ignoriert.")
    if mc_mode == "nur_sparplan" and common_start_cap > 0:
        st.warning("Du hast 'Nur Sparrate' gewählt, aber Startkapital > 0 gesetzt. "
                "Das ist ok, aber das Startkapital wird im MC ignoriert.")
        
    run_mc = st.button("Monte Carlo starten", key="run_mc")

    if run_mc:
        try:
            with st.spinner("Lade historische Kursdaten…"):
                base_prices = load_prices(base_ticker.strip(), start, end)

            cfg = BacktestConfig(
                start_capital=float(common_start_cap),
                monthly_contribution=float(common_monthly),
                contrib_day=int(common_day),
                slippage_bps=float(common_slippage),
                trading_days=int(tdays),
            )

            mc_cfg = MonteCarloConfig(
                horizon_years=int(mc_horizon),
                n_paths=int(mc_paths),
                mode=mc_mode,
            )

            with st.spinner("Simuliere Zukunftspfade…"):
                mc_res = run_mc_bootstrap(
                    base_prices_hist=base_prices,
                    cfg=cfg,
                    mc=mc_cfg,
                    leverage=float(leverage),
                    annual_fin=float(annual_fin),
                    annual_fee=float(annual_fee),
                )

            st.subheader("Ergebnis – Verteilung der Endwerte")
            st.dataframe(
                mc_res.summary.style.format({
                    "Investiert (Modus)": "{:,.2f}",
                    "Ø Endwert": "{:,.2f}",
                    "Median Endwert": "{:,.2f}",
                    "5%-Perzentil": "{:,.2f}",
                    "95%-Perzentil": "{:,.2f}",
                    "P(Endwert < Basis)": "{:.1%}",
                })
            )

            st.subheader("Zeitverlauf – Median & 5–95%-Band")
            base_band = mc_res.bands["base"]
            syn_band = mc_res.bands["syn"]

            chart = pd.DataFrame({
                "Basis p50": base_band["p50"],
                "Basis p5": base_band["p5"],
                "Basis p95": base_band["p95"],
                f"Synth {leverage:.1f}x p50": syn_band["p50"],
                f"Synth {leverage:.1f}x p5": syn_band["p5"],
                f"Synth {leverage:.1f}x p95": syn_band["p95"],
            })
            st.line_chart(chart)

            st.subheader("Downloads – Endwerte (je Pfad)")
            end_df = pd.DataFrame({
                "end_base": mc_res.end_values["base"],
                "end_synth": mc_res.end_values["syn"],
            })
            st.download_button(
                "CSV Endwerte",
                end_df.to_csv(index=False).encode("utf-8"),
                file_name="mc_endwerte.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(str(e))
