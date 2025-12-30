import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


from engine import (
    BacktestConfig,
    MonteCarloConfig,
    run_dca_backtest,
    make_synthetic_leverage_prices,
    run_mc_bootstrap,
    TRADING_DAYS_DEFAULT,
)

from metrics import (
    calc_metrics_simple,
    calc_drawdown,
    xirr_from_cashflows,
    make_cashflows_from_contrib_and_final,
    time_to_recovery,
    rolling_loss_share,
    rolling_loss_stats,
    rolling_underperformance_share,
    rolling_underperformance_stats,
    rolling_severe_loss_share
)

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

# -------------------------------------------------
# Helper Funktionen: Grafiken
# -------------------------------------------------

def plot_line_years(df: pd.DataFrame, title: str, y_title: str = ""):
    # df: Index = DatetimeIndex, Spalten = Linien
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=str(col)))

    fig.update_layout(
        title=title,
        xaxis_title="Jahr",
        yaxis_title=y_title,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    # Nur Jahre anzeigen
    fig.update_xaxes(
        tickformat="%Y",
        dtick="M12",       # jedes Jahr ein Tick
        showgrid=True
    )
    fig.update_yaxes(showgrid=True)

    # Wichtig: kein Scroll-Zoom + Achsen fix (kein Zoom/Pan)
    st.plotly_chart(
        fig,
        width="stretch",
        config={"scrollZoom": False, "displayModeBar": True},
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

def drawdown_events(equity: pd.Series):
    """
    Liefert Ereignisse zum maximalen Drawdown:
    - peak_date: Datum des vorherigen ATH (Beginn des MaxDD)
    - trough_date: Datum des maximalen Drawdowns
    - recovery_date: erstes Datum, an dem Equity wieder >= altes ATH ist (oder None)
    - maxdd: maximaler Drawdown (negativ)
    - recovery_days: Tage von peak_date bis recovery_date (oder None)
    """
    eq = equity.astype(float).dropna()
    if len(eq) < 3:
        return {"peak_date": None, "trough_date": None, "recovery_date": None, "maxdd": np.nan, "recovery_days": None}

    peak = eq.cummax()
    dd = eq / peak - 1.0
    if dd.empty:
        return {"peak_date": None, "trough_date": None, "recovery_date": None, "maxdd": np.nan, "recovery_days": None}

    trough_date = dd.idxmin()
    maxdd = float(dd.loc[trough_date])

    peak_to_trough = peak.loc[:trough_date]
    if peak_to_trough.empty:
        return {"peak_date": None, "trough_date": trough_date, "recovery_date": None, "maxdd": maxdd, "recovery_days": None}

    peak_value = float(peak_to_trough.max())
    peak_date = peak_to_trough.idxmax()

    after = eq.loc[trough_date:]
    rec = after[after >= peak_value]
    if rec.empty:
        return {"peak_date": peak_date, "trough_date": trough_date, "recovery_date": None, "maxdd": maxdd, "recovery_days": None}

    recovery_date = rec.index[0]
    recovery_days = int((recovery_date - peak_date).days)

    return {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "maxdd": maxdd,
        "recovery_days": recovery_days,
    }

def add_vline_dt(fig, x_dt, color, dash, label=None, side="left"):
    """
    Vertikale Linie für Datetime-Achse (robust ohne Plotly add_vline / Pandas Timestamp-Bug).
    side: 'left' oder 'right' (Annotation Position)
    """
    if x_dt is None:
        return

    # Pandas Timestamp -> Python datetime
    x_py = pd.to_datetime(x_dt).to_pydatetime()

    # Vertikale Linie über die gesamte Plot-Höhe
    fig.add_shape(
        type="line",
        x0=x_py, x1=x_py,
        y0=0, y1=1,
        xref="x",
        yref="paper",
        line=dict(color=color, width=2, dash=dash),
    )

    # Optional: Text oben an die Linie
    if label:
        fig.add_annotation(
            x=x_py,
            y=1,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            yanchor="bottom",
            xanchor=("left" if side == "left" else "right"),
            font=dict(color=color),
        )

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

    e1, e2, _ = st.columns([1, 1, 2])

    with e1:
        rf = st.number_input(
            "Risikofrei p.a. (%)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key="bt_rf",
            help="Wird für Sharpe genutzt (annualisiert)."
        ) / 100.0

    with e2:
        rolling_years = st.number_input(
            "Rolling-Horizont (Jahre)",
            min_value=1,
            max_value=30,
            value=5,
            step=1,
            key="bt_roll_years",
            help="Fensterlänge für Rolling-Auswertungen (z.B. 1, 3, 5 oder 10 Jahre)."
        )

    
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

            st.divider()

            st.subheader("Kennzahlen")

            st.markdown("#### At a Glance")   

            c1, c2 = st.columns(2)
            c1.metric("Basis Endwert", f"{base_res.summary['final_value']:,.2f}")
            c1.metric("Basis P/L %", f"{base_res.summary['pl_pct']:,.2f}%")
            c2.metric(f"Synth {leverage:.1f}x Endwert", f"{syn_res.summary['final_value']:,.2f}")
            c2.metric("Synth P/L %", f"{syn_res.summary['pl_pct']:,.2f}%")

            # --- Basis Kennzahlen (inkl. Calmar aus metrics.py) ---
            m_base = calc_metrics_simple(base_res.equity, rf_annual=rf, trading_days=int(tdays))
            m_syn = calc_metrics_simple(syn_res.equity, rf_annual=rf, trading_days=int(tdays))

            # --- Neue Kennzahlen ---
            # TTR in Jahren (None -> NaN)
            ttr_base_days = time_to_recovery(base_res.equity)
            ttr_syn_days = time_to_recovery(syn_res.equity)

            ttr_base_years = (ttr_base_days / 365.25) if ttr_base_days is not None else np.nan
            ttr_syn_years = (ttr_syn_days / 365.25) if ttr_syn_days is not None else np.nan

            # Rolling Loss Share + Anzahl Fenster
            loss_share_base, n_windows = rolling_loss_stats(
                base_res.equity,
                window_years=int(rolling_years),
                trading_days=int(tdays),
            )

            loss_share_syn, _ = rolling_loss_stats(
                syn_res.equity,
                window_years=int(rolling_years),
                trading_days=int(tdays),
            )

            underperf_share_syn = rolling_underperformance_share(
                lev_equity=syn_res.equity,
                base_equity=base_res.equity,
                window_years=int(rolling_years),
                trading_days=int(tdays),
            )

            # ---------- Underperformance-Stat einmalig (statt doppelt) ----------
            under_share, n_windows_u = rolling_underperformance_stats(
                lev_equity=syn_res.equity,
                base_equity=base_res.equity,
                window_years=int(rolling_years),
                trading_days=int(tdays),
            )

            # ---------- Werte extrahieren ----------
            base_final    = float(base_res.summary.get("final_value", np.nan))
            syn_final     = float(syn_res.summary.get("final_value", np.nan))
            base_invested = float(base_res.summary.get("invested", np.nan))
            syn_invested  = float(syn_res.summary.get("invested", np.nan))
            base_pl_pct   = float(base_res.summary.get("pl_pct", np.nan))
            syn_pl_pct    = float(syn_res.summary.get("pl_pct", np.nan))

            base_cagr = float(m_base.get("CAGR", np.nan)) * 100.0
            syn_cagr  = float(m_syn.get("CAGR", np.nan)) * 100.0

            base_vol   = float(m_base.get("Vol", np.nan)) * 100.0
            syn_vol    = float(m_syn.get("Vol", np.nan)) * 100.0
            base_maxdd = float(m_base.get("MaxDD", np.nan)) * 100.0
            syn_maxdd  = float(m_syn.get("MaxDD", np.nan)) * 100.0
            base_calmar = float(m_base.get("Calmar", np.nan))
            syn_calmar  = float(m_syn.get("Calmar", np.nan))

            base_ttr = float(ttr_base_years) if pd.notna(ttr_base_years) else np.nan
            syn_ttr  = float(ttr_syn_years) if pd.notna(ttr_syn_years) else np.nan

            base_loss = float(loss_share_base) * 100.0 if pd.notna(loss_share_base) else np.nan
            syn_loss  = float(loss_share_syn) * 100.0 if pd.notna(loss_share_syn) else np.nan

            under_vs_1x = float(under_share) * 100.0 if pd.notna(under_share) else np.nan

            # Severe Loss Share (z. B. < -20 %)
            SEVERE_LOSS_THRESHOLD = -0.20

            severe_loss_base = rolling_severe_loss_share(
                base_res.equity,
                window_years=int(rolling_years),
                loss_threshold=SEVERE_LOSS_THRESHOLD,
                trading_days=int(tdays),
            )

            severe_loss_syn = rolling_severe_loss_share(
                syn_res.equity,
                window_years=int(rolling_years),
                loss_threshold=SEVERE_LOSS_THRESHOLD,
                trading_days=int(tdays),
            )

            # =========================
            # Kennzahlen
            # =========================

            # --- Renditekennzahlen ---
            rendite = pd.DataFrame(
                {
                    "Endwert": [base_res.summary["final_value"], syn_res.summary["final_value"]],
                    "Investiert": [base_res.summary["invested"], syn_res.summary["invested"]],
                    "Gesamtperformance (%)": [base_res.summary["pl_pct"], syn_res.summary["pl_pct"]],
                    "CAGR": [m_base.get("CAGR", np.nan), m_syn.get("CAGR", np.nan)],
                },
                index=[f"Basis {base_ticker}", f"Synth {leverage:.1f}x"],
            )

            st.markdown("#### Rendite")
            st.dataframe(
                rendite.style.format(
                    {
                        "Endwert": "{:,.2f}",
                        "Investiert": "{:,.2f}",
                        "Gesamtperformance (%)": "{:.2f}%",
                        "CAGR": "{:.2%}",
                    }
                ),
                width="stretch",
            )

            # --- Risiko: Pfad ---
            risiko_pfad = pd.DataFrame(
                {
                    "Volatilität": [m_base.get("Vol", np.nan), m_syn.get("Vol", np.nan)],
                    "Max Drawdown": [m_base.get("MaxDD", np.nan), m_syn.get("MaxDD", np.nan)],
                    "Time to Recovery (Jahre)": [ttr_base_years, ttr_syn_years],
                    "Calmar Ratio": [m_base.get("Calmar", np.nan), m_syn.get("Calmar", np.nan)],
                },
                index=[f"Basis {base_ticker}", f"Synth {leverage:.1f}x"],
            )

            st.markdown("#### Risiko - Pfad")
            st.dataframe(
                risiko_pfad.style.format(
                    {
                        "Volatilität": "{:.2%}",
                        "Max Drawdown": "{:.2%}",
                        "Time to Recovery (Jahre)": "{:.2f}",
                        "Calmar Ratio": "{:.2f}",
                    }
                ),
                width="stretch",
            )
            
            # -------------------------
            # Risiko – Rolling Window            
            risiko_rolling = pd.DataFrame(
                {
                    f"Loss Share ({int(rolling_years)}J)": [
                        loss_share_base,
                        loss_share_syn,
                    ],
                    f"Severe Loss Share < {int(abs(SEVERE_LOSS_THRESHOLD)*100)}% ({int(rolling_years)}J)": [
                        severe_loss_base,
                        severe_loss_syn,
                    ],
                    f"Underperformance vs 1x ({int(rolling_years)}J)": [
                        np.nan,
                        underperf_share_syn,
                    ],
                },
                index=[f"Basis {base_ticker}", f"Synth {leverage:.1f}x"],
            )

            st.markdown(f"#### Risiko - Rolling Window {int(rolling_years)} Jahr(e)")
            st.dataframe(
                risiko_rolling.style.format(
                    {
                        f"Loss Share ({int(rolling_years)}J)": "{:.1%}",
                        f"Severe Loss Share < {int(abs(SEVERE_LOSS_THRESHOLD)*100)}% ({int(rolling_years)}J)": "{:.1%}",
                        f"Underperformance vs 1x ({int(rolling_years)}J)": "{:.1%}",
                    }
                ),
                width="stretch",
            )


            # -------------------------------
            # Chart Equity + Einzahlungen
            
            st.divider()
            
            st.subheader("Portfoliowert & Einzahlungen")
            df_eq = pd.DataFrame({
                f"Basis {base_ticker}": base_res.equity,
                f"Synth {leverage:.1f}x": syn_res.equity,
                "Einzahlungen (kum.)": base_res.contrib,
            }).dropna(how="all")

            plot_line_years(df_eq, title=" ", y_title="Wert")

            # -------------------------------
            # Chart Drawdowns

            st.subheader("Drawdowns & Time to Recovery")

            COLOR_BASE = "#6e6e6e"        # dunkelgrau
            COLOR_LETF = "#1f77b4"        # plotly-blau

            COLOR_BASE_MARK = "#c7c7c7"   # hellgrau
            COLOR_LETF_MARK = "#9ecae1"   # hellblau


            dd_base = calc_drawdown(base_res.equity)
            dd_syn = calc_drawdown(syn_res.equity)

            ev_base = drawdown_events(base_res.equity)
            ev_syn = drawdown_events(syn_res.equity)

            fig_dd = go.Figure()

            # Linien
            fig_dd.add_trace(go.Scatter(
                x=dd_base.index,
                y=dd_base.values,
                mode="lines",
                name=f"Basis {base_ticker}",
                line=dict(color=COLOR_BASE, width=2),
            ))

            fig_dd.add_trace(go.Scatter(
                x=dd_syn.index,
                y=dd_syn.values,
                mode="lines",
                name=f"Synth {leverage:.1f}x",
                line=dict(color=COLOR_LETF, width=2),
            ))

            # Basis – Max Drawdown
            if ev_base["trough_date"] is not None:
                add_vline_dt(
                    fig_dd,
                    ev_base["trough_date"],
                    color=COLOR_BASE_MARK,
                    dash="dot",
                    label="MaxDD",
                    side="left",
                )

            # Basis – Recovery
            if ev_base["recovery_date"] is not None:
                yrs = (ev_base["recovery_days"] / 365.25) if (ev_base["recovery_days"] is not None) else None
                label = f"Recovery ({yrs:.1f} J)" if yrs is not None else "Recovery"
                add_vline_dt(
                    fig_dd,
                    ev_base["recovery_date"],
                    color=COLOR_BASE_MARK,
                    dash="dash",
                    label=label,
                    side="left",
                )

            # LETF – Max Drawdown
            if ev_syn["trough_date"] is not None:
                add_vline_dt(
                    fig_dd,
                    ev_syn["trough_date"],
                    color=COLOR_LETF_MARK,
                    dash="dot",
                    label="MaxDD",
                    side="right",
                )

            # LETF – Recovery
            if ev_syn["recovery_date"] is not None:
                yrs = (ev_syn["recovery_days"] / 365.25) if (ev_syn["recovery_days"] is not None) else None
                label = f"Recovery ({yrs:.1f} J)" if yrs is not None else "Recovery"
                add_vline_dt(
                    fig_dd,
                    ev_syn["recovery_date"],
                    color=COLOR_LETF_MARK,
                    dash="dash",
                    label=label,
                    side="right",
                )

            # Layout: nur Jahre auf der X-Achse, kein Zoom per Scroll
            fig_dd.update_layout(
                title=" ",
                xaxis_title="Jahr",
                yaxis_title="Drawdown",
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            fig_dd.update_xaxes(tickformat="%Y", dtick="M12", showgrid=True)
            fig_dd.update_yaxes(showgrid=True)

            st.plotly_chart(fig_dd, width="stretch", config={"scrollZoom": False, "displayModeBar": True})

            # Info-Boxen: MaxDD + Recovery-Zeit
            def _fmt_event(name: str, ev: dict):
                if ev["trough_date"] is None or np.isnan(ev["maxdd"]):
                    return f"**{name}:** Keine ausreichenden Daten."
                maxdd_pct = ev["maxdd"] * 100
                t = ev["trough_date"].date()
                if ev["recovery_date"] is None:
                    return f"**{name}:** MaxDD **{maxdd_pct:.1f}%** am **{t}** · **Kein Recovery** bis zum Periodenende."
                r = ev["recovery_date"].date()
                days = ev["recovery_days"]
                yrs = days / 365.25 if days is not None else None
                dur = f"{days} Tage (~{yrs:.1f} Jahre)" if yrs is not None else f"{days} Tage"
                return f"**{name}:** MaxDD **{maxdd_pct:.1f}%** am **{t}** · Recovery am **{r}** · Dauer: **{dur}**"

            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info(_fmt_event(f"Basis {base_ticker}", ev_base))
            with col_info2:
                st.info(_fmt_event(f"Synth {leverage:.1f}x", ev_syn))

            # -------------------------------
            # Loss Share Bar Chart

            st.subheader("Rolling Window: Anteil der Verlust-Perioden")

            loss_base = loss_share_base * 100
            loss_syn = loss_share_syn * 100

            df_bar = pd.DataFrame({
                "Produkt": [f"Basis {base_ticker}", f"Synth {leverage:.1f}x"],
                "Anteil Verlust-Fenster (%)": [loss_base, loss_syn],
            })

            fig = px.bar(
                df_bar,
                x="Produkt",
                y="Anteil Verlust-Fenster (%)",
                text="Anteil Verlust-Fenster (%)",
                title=" ",
            )

            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

            fig.update_layout(
                yaxis_range=[0, 100],
                yaxis_title="Anteil der Fenster mit negativem Ergebnis (%)",
                xaxis_title="",
                margin=dict(l=10, r=10, t=80, b=10),
            )


            fig.add_annotation(
                text=f"Anzahl der berücksichtigten Rolling-Fenster: {n_windows}",
                xref="paper",
                yref="paper",
                x=0,
                y=1.08,
                showarrow=False,
                align="left",
                font=dict(size=12, color="gray"),
            )

            st.plotly_chart(fig, width="stretch", config={"scrollZoom": False, "displayModeBar": True})

            col_info_base, col_info_syn = st.columns(2)

            if loss_share_base == 0:
                with col_info_base:
                    st.info(
                        f"**Basis {base_ticker}:** "
                        f"In keinem der {n_windows} Rolling-Fenster ({rolling_years} Jahre) trat ein Verlust auf."
                    )

            if loss_share_syn == 0:
                with col_info_syn:
                    st.info(
                        f"**Synth {leverage:.1f}x:** "
                        f"In keinem der {n_windows} Rolling-Fenster ({rolling_years} Jahre) trat ein Verlust auf."
                    )

            # -------------------------------
            # Chart Rolling Underperformance Share

            under_pct = under_share * 100 if pd.notna(under_share) else np.nan

            st.subheader("Rolling Vergleich: LETF schlechter als 1x")

            df_under = pd.DataFrame({
                "Vergleich": [f"Synth {leverage:.1f}x vs. Basis {base_ticker}"],
                "Anteil Fenster (LETF < 1x) (%)": [under_pct],
            })

            fig_u = px.bar(
                df_under,
                x="Vergleich",
                y="Anteil Fenster (LETF < 1x) (%)",
                text="Anteil Fenster (LETF < 1x) (%)",
                title=" ",
            )

            fig_u.update_traces(texttemplate="%{text:.1f}%", textposition="outside")

            fig_u.update_layout(
                yaxis_range=[0, 100],
                yaxis_title="Anteil der Rolling-Fenster (%)",
                xaxis_title="",
                margin=dict(l=10, r=10, t=80, b=10),
            )

            fig_u.add_annotation(
                text=f"Anzahl der berücksichtigten Rolling-Fenster: {n_windows_u}",
                xref="paper",
                yref="paper",
                x=0,
                y=1.08,
                showarrow=False,
                align="left",
                font=dict(size=12, color="gray"),
            )

            st.plotly_chart(
                fig_u,
                width="stretch",
                config={"scrollZoom": False, "displayModeBar": True},
            )

            # Optional: Hinweis bei 0%
            if pd.notna(under_share) and under_share == 0:
                st.info(
                    f"**Synth {leverage:.1f}x:** In keinem der {n_windows_u} Rolling-Fenster "
                    f"({rolling_years} Jahre) schnitt der LETF schlechter ab als der ungehebelte ETF (1x)."
                )


            # -------------------------------
            # Download Buttons


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

            st.divider()

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

            # Basis
            df_base = pd.DataFrame({
                "Basis p50": base_band["p50"],
                "Basis p5": base_band["p5"],
                "Basis p95": base_band["p95"],
            }).dropna(how="all")
            plot_line_years(df_base, title="Monte Carlo – Basis (Median & Band)", y_title="Portfoliowert")

            # Synth
            df_syn = pd.DataFrame({
                f"Synth {leverage:.1f}x p50": syn_band["p50"],
                f"Synth {leverage:.1f}x p5": syn_band["p5"],
                f"Synth {leverage:.1f}x p95": syn_band["p95"],
            }).dropna(how="all")
            plot_line_years(df_syn, title=f"Monte Carlo – Synth {leverage:.1f}x (Median & Band)", y_title="Portfoliowert")


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
