# ============================
# backtest.py  (NEU)
# ============================
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from engine import run_dca_backtest, make_synthetic_leverage_prices
from metrics import (
    calc_metrics_simple,
    calc_drawdown,
    time_to_recovery,
    rolling_loss_stats,
    rolling_underperformance_share,
    rolling_underperformance_stats,
    rolling_severe_loss_share,
)

# -----------------------
# Helper: Drawdown Events
# -----------------------
def drawdown_events(equity: pd.Series):
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


def add_vline_dt(
    fig,
    x_dt,
    color,
    dash,
    label=None,
    side="left",
    y_pos=1.0,
    y_shift=0,
    x_shift=None,
):
    if x_dt is None:
        return

    x_py = pd.to_datetime(x_dt).to_pydatetime()

    fig.add_shape(
        type="line",
        x0=x_py, x1=x_py,
        y0=0, y1=1,
        xref="x",
        yref="paper",
        line=dict(color=color, width=2, dash=dash),
    )

    if not label:
        return

    if x_shift is None:
        x_shift = -25 if side == "left" else 25

    fig.add_annotation(
        x=x_py,
        y=y_pos,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        yanchor="bottom",
        xanchor=("right" if side == "left" else "left"),
        font=dict(color=color),
        yshift=y_shift,
        xshift=x_shift,
    )


def render_backtest_tab(
    *,
    load_prices,
    base_ticker: str,
    start: str,
    end: str,
    cfg,
    leverage: float,
    annual_fin: float,
    annual_fee: float,
    tdays: int,
    color_map: dict,
):
    st.subheader("Backtesting")

    st.caption(
        "Das Backtesting analysiert die historische Wertentwicklung eines Basis-ETFs "
        "und eines synthetisch modellierten gehebelten ETFs. "
        "Rolling-Window-Analysen reduzieren die Abhängigkeit vom Einstiegszeitpunkt."
    )


    st.markdown("#### Einstellungen")

    e1, _ = st.columns([1, 3])

    with e1:
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

    if not run_bt:
        return

    try:
        with st.spinner("Lade Kursdaten…"):
            base_prices = load_prices(base_ticker.strip(), start, end)

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
        c1.metric("Basis Endwert", f"{base_res.summary['final_value']:,.0f}")
        c1.metric("Basis P/L %", f"{base_res.summary['pl_pct']:,.1f}%")
        c2.metric(f"Synth {leverage:.1f}x Endwert", f"{syn_res.summary['final_value']:,.0f}")
        c2.metric("Synth P/L %", f"{syn_res.summary['pl_pct']:,.1f}%")

        # ---------- Kennzahlen ----------
        m_base = calc_metrics_simple(base_res.equity, trading_days=int(tdays))
        m_syn = calc_metrics_simple(syn_res.equity, trading_days=int(tdays))

        ttr_base_days = time_to_recovery(base_res.equity)
        ttr_syn_days = time_to_recovery(syn_res.equity)
        ttr_base_years = (ttr_base_days / 365.25) if ttr_base_days is not None else np.nan
        ttr_syn_years = (ttr_syn_days / 365.25) if ttr_syn_days is not None else np.nan

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

        under_share, n_windows_u = rolling_underperformance_stats(
            lev_equity=syn_res.equity,
            base_equity=base_res.equity,
            window_years=int(rolling_years),
            trading_days=int(tdays),
        )

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

        # Rendite
        rendite = pd.DataFrame(
            {
                "Endwert": [base_res.summary["final_value"], syn_res.summary["final_value"]],
                "Investiert": [base_res.summary["invested"], syn_res.summary["invested"]],
                "Gesamtperformance": [base_res.summary["pl_pct"], syn_res.summary["pl_pct"]],
                "CAGR": [m_base.get("CAGR", np.nan), m_syn.get("CAGR", np.nan)],
            },
            index=[f"Basis {base_ticker}", f"Synth {leverage:.1f}x"],
        )

        st.markdown("#### Rendite")
        st.dataframe(
            rendite.style.format(
                {
                    "Endwert": "{:,.0f}",
                    "Investiert": "{:,.0f}",
                    "Gesamtperformance": "{:.1f}%",
                    "CAGR": "{:.1%}",
                }
            ),
            width="stretch",
        )

        # Risiko – Pfad
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
                    "Volatilität": "{:.1%}",
                    "Max Drawdown": "{:.1%}",
                    "Time to Recovery (Jahre)": "{:.2f}",
                    "Calmar Ratio": "{:.2f}",
                }
            ),
            width="stretch",
        )

        # Risiko – Rolling
        risiko_rolling = pd.DataFrame(
            {
                f"Loss Share ({int(rolling_years)}J)": [loss_share_base, loss_share_syn],
                f"Severe Loss Share < {int(abs(SEVERE_LOSS_THRESHOLD)*100)}% ({int(rolling_years)}J)": [
                    severe_loss_base, severe_loss_syn
                ],
                f"Underperformance vs 1x ({int(rolling_years)}J)": [np.nan, underperf_share_syn],
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
        # Charts
        st.divider()
        st.subheader("Portfoliowert & Einzahlungen")

        df_eq = pd.DataFrame({
            f"Basis {base_ticker}": base_res.equity,
            f"Synth {leverage:.1f}x": syn_res.equity,
            "Einzahlungen (kum.)": base_res.contrib,
        }).dropna(how="all")

        fig = go.Figure()
        for col in df_eq.columns:
            if "Basis" in col:
                color = color_map["Basis"]
            elif "Synth" in col:
                color = color_map["Synth"]
            else:
                color = color_map["Einzahlungen"]

            fig.add_trace(go.Scatter(
                x=df_eq.index,
                y=df_eq[col],
                mode="lines",
                name=str(col),
                line=dict(color=color, width=2),
            ))

        fig.update_layout(
            xaxis_title="Jahr",
            yaxis_title="Wert",
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig.update_xaxes(tickformat="%Y", dtick="M12", showgrid=True)
        fig.update_yaxes(showgrid=True)

        st.plotly_chart(fig, width="stretch", config={"scrollZoom": False})

        # Drawdowns
        st.subheader("Drawdowns & Time to Recovery")

        COLOR_BASE = color_map["Basis"]
        COLOR_LETF = color_map["Synth"]

        dd_base = calc_drawdown(base_res.equity)
        dd_syn = calc_drawdown(syn_res.equity)

        ev_base = drawdown_events(base_res.equity)
        ev_syn = drawdown_events(syn_res.equity)

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dd_base.index, y=dd_base.values, mode="lines",
            name=f"Basis {base_ticker}",
            line=dict(color=COLOR_BASE, width=2),
        ))
        fig_dd.add_trace(go.Scatter(
            x=dd_syn.index, y=dd_syn.values, mode="lines",
            name=f"Synth {leverage:.1f}x",
            line=dict(color=COLOR_LETF, width=2),
        ))

        # Basis Markierungen
        if ev_base["trough_date"] is not None:
            add_vline_dt(fig_dd, ev_base["trough_date"], color=COLOR_BASE, dash="dot", label="MaxDD", side="left", y_pos=1.06)
        if ev_base["recovery_date"] is not None:
            yrs = (ev_base["recovery_days"] / 365.25) if ev_base["recovery_days"] is not None else None
            label = f"Recovery ({yrs:.1f} J)" if yrs is not None else "Recovery"
            add_vline_dt(fig_dd, ev_base["recovery_date"], color=COLOR_BASE, dash="dash", label=label, side="right", y_pos=1.06)

        # LETF Markierungen
        if ev_syn["trough_date"] is not None:
            add_vline_dt(fig_dd, ev_syn["trough_date"], color=COLOR_LETF, dash="dot", label="MaxDD", side="left", y_pos=1.02)
        if ev_syn["recovery_date"] is not None:
            yrs = (ev_syn["recovery_days"] / 365.25) if ev_syn["recovery_days"] is not None else None
            label = f"Recovery ({yrs:.1f} J)" if yrs is not None else "Recovery"
            add_vline_dt(fig_dd, ev_syn["recovery_date"], color=COLOR_LETF, dash="dash", label=label, side="right", y_pos=1.02)

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

        # Loss Share Bar
        st.subheader("Rolling Window: Anteil der Verlust-Perioden")

        loss_base = loss_share_base * 100
        loss_syn = loss_share_syn * 100

        df_bar = pd.DataFrame({
            "Produkt": [f"Basis {base_ticker}", f"Synth {leverage:.1f}x"],
            "Anteil Verlust-Fenster (%)": [loss_base, loss_syn],
        })

        bar_colors = [color_map["Basis"], color_map["Synth"]]

        fig_bar = px.bar(
            df_bar,
            x="Produkt",
            y="Anteil Verlust-Fenster (%)",
            text="Anteil Verlust-Fenster (%)",
            title=" ",
        )

        fig_bar.update_traces(
            marker_color=bar_colors,
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        fig_bar.update_layout(
            yaxis_range=[0, 100],
            yaxis_title="Anteil der Fenster mit negativem Ergebnis (%)",
            xaxis_title="",
            margin=dict(l=10, r=10, t=80, b=10),
        )

        fig_bar.add_annotation(
            text=f"Anzahl der berücksichtigten Rolling-Fenster: {n_windows}",
            xref="paper",
            yref="paper",
            x=0,
            y=1.08,
            showarrow=False,
            align="left",
            font=dict(size=12, color="gray"),
        )

        st.plotly_chart(fig_bar, width="stretch", config={"scrollZoom": False, "displayModeBar": True})

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

        # Downloads
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
