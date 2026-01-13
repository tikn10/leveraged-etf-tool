# ============================
# mc.py
# ============================
import pandas as pd
import streamlit as st

from engine import MonteCarloConfig, run_mc_bootstrap


def plot_mc_combined(
    *,
    base_band: dict,
    syn_band: dict,
    leverage: float,
    title: str,
    y_title: str = "Portfoliowert",
):
    import plotly.graph_objects as go

    # Farben (NEU)
    COLOR_BASE_LINE = "#6e6e6e"                 # grau
    COLOR_BASE_BAND = "rgba(110,110,110,0.25)"  # hellgrau

    COLOR_SYN_LINE  = "#1f77b4"                 # blau
    COLOR_SYN_BAND  = "rgba(31,119,180,0.25)"   # hellblau

    fig = go.Figure()

    # Base band (filled)
    fig.add_trace(go.Scatter(
        x=base_band["p_high"].index,
        y=base_band["p_high"].values,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=base_band["p_low"].index,
        y=base_band["p_low"].values,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=COLOR_BASE_BAND,   # <-- NEU
        name="Basis Konfidenzband",
    ))
    fig.add_trace(go.Scatter(
        x=base_band["p50"].index,
        y=base_band["p50"].values,
        mode="lines",
        line=dict(color=COLOR_BASE_LINE, width=2),  # <-- NEU
        name="Basis Median",
    ))

    # Synth band (filled)
    fig.add_trace(go.Scatter(
        x=syn_band["p_high"].index,
        y=syn_band["p_high"].values,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=syn_band["p_low"].index,
        y=syn_band["p_low"].values,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=COLOR_SYN_BAND,   # <-- NEU
        name=f"Synth {leverage:.1f}x Konfidenzband",
    ))
    fig.add_trace(go.Scatter(
        x=syn_band["p50"].index,
        y=syn_band["p50"].values,
        mode="lines",
        line=dict(color=COLOR_SYN_LINE, width=2),  # <-- NEU
        name=f"Synth {leverage:.1f}x Median",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Jahr",
        yaxis_title=y_title,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(tickformat="%Y", dtick="M12", showgrid=True)
    fig.update_yaxes(showgrid=True)

    st.plotly_chart(fig, width="stretch", config={"scrollZoom": False, "displayModeBar": True},)


def render_mc_tab(
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
    color_map: dict,  # aktuell nicht genutzt, aber bewusst schon übergeben
):
    st.subheader("Monte Carlo Simulation")

    st.caption(
        "Die Monte-Carlo-Simulation erzeugt auf Basis historischer Tagesrenditen "
        "eine Vielzahl möglicher Zukunftspfade. "
        "Die Ergebnisse sind probabilistisch zu interpretieren."
    )

    st.markdown("#### Einstellungen")

    f1, f2, f3, f4 = st.columns([1, 1, 1, 1])

    with f1:
        mc_horizon = st.number_input(
            "Horizont (Jahre)",
            min_value=1,
            max_value=40,
            value=10,
            step=1,
            key="mc_h",
            help="Wie viele Jahre ab dem letzten historischen Kurstag simuliert werden."
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
    with f4:
        ci_label = st.selectbox(
            "Konfidenzintervall",
            ["95%", "90%"],
            index=0,
            key="mc_ci",
            help="Steuert Perzentile und VaR/ES-Quantil (95% -> 5%, 90% -> 10%)."
        )

    if mc_mode_label == "Startkapital + Sparrate":
        mc_mode = "start_und_sparplan"
    elif mc_mode_label == "Nur Sparrate":
        mc_mode = "nur_sparplan"
    else:
        mc_mode = "nur_start"

    ci_level = 0.95 if ci_label == "95%" else 0.90

    if mc_mode == "nur_start" and cfg.monthly_contribution > 0:
        st.warning("Du hast 'Nur Startkapital' gewählt, aber eine Sparrate > 0 gesetzt. "
                   "Das ist ok, aber die Sparrate wird im MC ignoriert.")
    if mc_mode == "nur_sparplan" and cfg.start_capital > 0:
        st.warning("Du hast 'Nur Sparrate' gewählt, aber Startkapital > 0 gesetzt. "
                   "Das ist ok, aber das Startkapital wird im MC ignoriert.")

    run_mc = st.button("Monte Carlo starten", key="run_mc")
    if not run_mc:
        return

    try:
        with st.spinner("Lade historische Kursdaten…"):
            base_prices = load_prices(base_ticker.strip(), start, end)

        mc_cfg = MonteCarloConfig(
            horizon_years=int(mc_horizon),
            n_paths=int(mc_paths),
            mode=mc_mode,
            ci_level=float(ci_level),
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

        # --------------------------
        # Assumptions
        st.divider()
        st.subheader("Annahmen")

        a = mc_res.assumptions

        col_left, col_right = st.columns([1, 2])

        # Linke Spalte: Historie
        with col_left:
            st.markdown("**Historische Daten**")
            st.write(f"Zeitraum: {a['historische_daten']['Zeitraum']}")
            st.write(f"Tage: {a['historische_daten']['Tage']}")

        # Rechte Spalte: Tabelle
        with col_right:
            st.markdown("**Historische Rendite- & Risikoparameter**")

            df_params = pd.DataFrame(
                {
                    "Rendite p.a. (historisch)": [
                        a["basis_etf"]["Rendite p.a. (implizit)"],
                        a["synth_etf"]["Rendite p.a. (implizit)"],
                    ],
                    "Volatilität p.a. (historisch)": [
                        a["basis_etf"]["Volatilität p.a."],
                        a["synth_etf"]["Volatilität p.a."],
                    ],
                },
                index=["Basis-ETF", f"Synth {leverage:.1f}x"],
            )

            st.dataframe(
                df_params.style.format(
                    {
                        "Rendite p.a. (historisch)": "{:.1%}",
                        "Volatilität p.a. (historisch)": "{:.1%}",
                    }
                ),
                width="stretch",
            )

            st.caption(
                "Die dargestellten Parameter basieren auf historischen Tagesrenditen und "
                "dienen ausschließlich der Einordnung der Ergebnisse. "
                "Die Simulation basiert auf einem Bootstrap-Verfahren ohne parametrische Annahmen."
            )



        # --------------------------
        # Ergebnis – Kacheln

        st.divider()
        st.subheader("Kennzahlen")
        st.markdown("#### Rendite")

        # helper: unpack CI tuples in df
        def _get_ci_tuple(df, row, col):
            v = df.loc[row, col]
            if isinstance(v, tuple) and len(v) == 2:
                return v
            return (float("nan"), float("nan"))

        summ = mc_res.summary.set_index("Produkt")

        base_row = "Basis-ETF (MC)"
        syn_row = f"Synth {leverage:.1f}x (MC)"
        ci_end_col = [c for c in summ.columns if "CI Endwert" in c][0]
        ci_cagr_col = [c for c in summ.columns if "CI CAGR" in c][0]
        ci_tr_col = [c for c in summ.columns if "CI Gesamtperformance" in c][0]

        base_end_ci = _get_ci_tuple(summ, base_row, ci_end_col)
        syn_end_ci = _get_ci_tuple(summ, syn_row, ci_end_col)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Investiert", f"{float(summ.loc[base_row, 'Investiert (Modus)']):,.0f}")
        c2.metric("Median Endwert (Basis)", f"{float(summ.loc[base_row, 'Median Endwert']):,.0f}")
        c3.metric("Median Endwert (Synth)", f"{float(summ.loc[syn_row, 'Median Endwert']):,.0f}")
        c4.metric("P(Endwert Synth < Endwert Basis)", f"{float(summ.loc[syn_row, 'P(Endwert < Basis)']):.1%}")
        with c4:
            st.caption(
                "Anteil der simulierten Szenarien, in denen der Endwert des synthetischen ETFs "
                "unter dem Endwert des Basis-ETFs liegt."
            )

        # Tabellen mit Rendite-Kennzahlen

        df = mc_res.summary.copy()

        # 1) CI-Spalten dynamisch finden (passt zu 90% oder 95%)
        ci_end_col = next(c for c in df.columns if "CI" in c and "Endwert" in c)
        ci_cagr_col = next(c for c in df.columns if "CI" in c and "CAGR" in c)

        # Je nach Benennung: "Gesamtperformance" oder "Performance"
        perf_ci_candidates = [c for c in df.columns if "CI" in c and ("Gesamtperformance" in c or "Performance" in c)]
        ci_perf_col = perf_ci_candidates[0]

        # Formatter
        def fmt_money(x):
            return f"{float(x):,.0f}"

        def fmt_pct(x):
            return f"{float(x):.1%}"

        def fmt_ci_money(x):
            if isinstance(x, tuple) and len(x) == 2:
                return f"{x[0]:,.0f} – {x[1]:,.0f}"
            return ""

        def fmt_ci_pct(x):
            if isinstance(x, tuple) and len(x) == 2:
                return f"{x[0]:.1%} – {x[1]:.1%}"
            return ""
        
        def pretty_product(name: str) -> str:
            if name.startswith("Basis"):
                return f"Basis {base_ticker}"
            if name.startswith("Synth"):
                return name.replace(" (MC)", "")
            return name


        # -----------------------------
        # Tabelle 1: Investiert / Endwert
        # -----------------------------
        t1 = pd.DataFrame({
            "": df["Produkt"].apply(pretty_product),  
            "Investiert": df["Investiert (Modus)"],
            "Endwert (Median)": df["Median Endwert"],
            f"Endwert ({ci_end_col.replace(' Endwert','')})": df[ci_end_col],  # zeigt z.B. "5–95% CI"
        })

        t1["Investiert"] = t1["Investiert"].apply(fmt_money)
        t1["Endwert (Median)"] = t1["Endwert (Median)"].apply(fmt_money)
        t1[f"Endwert ({ci_end_col.replace(' Endwert','')})"] = t1[f"Endwert ({ci_end_col.replace(' Endwert','')})"].apply(fmt_ci_money)

        st.markdown("#### Endwerte")
        st.dataframe(t1.set_index(""), width="stretch")

        # -----------------------------
        # Tabelle 2: Performance / CAGR
        # -----------------------------
        st.markdown("#### Performance")

        t2 = pd.DataFrame({
            "": df["Produkt"].apply(pretty_product),  
            "CAGR (Median)": df["Median CAGR"],
            f"CAGR ({ci_cagr_col.replace(' CAGR','')})": df[ci_cagr_col],
            "Gesamtperformance (Median)": (
                df["Median Gesamtperformance"]
                if "Median Gesamtperformance" in df.columns
                else df["Median Performance"]
            ),
            f"Gesamtperformance ({ci_perf_col.replace(' Gesamtperformance','').replace(' Performance','')})": df[ci_perf_col],
        })

        # Formatierung: 1 Nachkommastelle
        t2["CAGR (Median)"] = t2["CAGR (Median)"].apply(fmt_pct)
        t2[f"CAGR ({ci_cagr_col.replace(' CAGR','')})"] = t2[
            f"CAGR ({ci_cagr_col.replace(' CAGR','')})"
        ].apply(fmt_ci_pct)

        t2["Gesamtperformance (Median)"] = t2["Gesamtperformance (Median)"].apply(fmt_pct)
        t2[f"Gesamtperformance ({ci_perf_col.replace(' Gesamtperformance','').replace(' Performance','')})"] = t2[
            f"Gesamtperformance ({ci_perf_col.replace(' Gesamtperformance','').replace(' Performance','')})"
        ].apply(fmt_ci_pct)

        st.dataframe(t2.set_index(""),width="stretch",)



        # --------------------------
        # Risiko – Wahrscheinlichkeiten & VaR/ES
        st.markdown("#### Risiko")


        def pretty_product(name: str) -> str:
            if name.startswith("Basis"):
                return f"Basis {base_ticker}"
            if name.startswith("Synth"):
                return name.replace(" (MC)", "")
            return name

        def fmt_pct_1(x):
            return f"{float(x):.1%}"


        rp = mc_res.risk_probs
        df = mc_res.summary.copy()

        risk_tbl = pd.DataFrame({
            "": df["Produkt"].apply(pretty_product),

            "Verlustwahrscheinlichkeit (Endwert < Einzahlungen)": [
                rp["loss_prob_base"],
                rp["loss_prob_syn"],
            ],

            f"Value at Risk ({ci_label})": [
                rp["var_base"],
                rp["var_syn"],
            ],

            f"Expected Shortfall ({ci_label})": [
                rp["es_base"],
                rp["es_syn"],
            ],
        })

        # Formatierung
        risk_tbl["Verlustwahrscheinlichkeit (Endwert < Einzahlungen)"] = risk_tbl["Verlustwahrscheinlichkeit (Endwert < Einzahlungen)"].apply(fmt_pct_1)
        risk_tbl[f"Value at Risk ({ci_label})"] = risk_tbl[f"Value at Risk ({ci_label})"].apply(fmt_pct_1)
        risk_tbl[f"Expected Shortfall ({ci_label})"] = risk_tbl[f"Expected Shortfall ({ci_label})"].apply(fmt_pct_1)

        st.dataframe(
            risk_tbl.set_index(""),
            width="stretch",
        )


        # MaxDD Wahrscheinlichkeiten
        st.markdown("**Wahrscheinlichkeit, dass der Max Drawdown mindestens … beträgt**")
        dd_base = rp["maxdd_probs_base"]
        dd_syn = rp["maxdd_probs_syn"]

        dd_df = pd.DataFrame({
            "Max Drawdown-Schwelle": [f"−{int(abs(k)*100)}%" for k in dd_base.keys()],
            f"Basis {base_ticker}": [dd_base[k] for k in dd_base.keys()],
            f"Synth {leverage:.1f}x": [dd_syn[k] for k in dd_syn.keys()],
        })

        st.dataframe(
            dd_df.set_index("Max Drawdown-Schwelle").style.format({
                f"Basis {base_ticker}": "{:.1%}",
                f"Synth {leverage:.1f}x": "{:.1%}",
            }),
            width="stretch",
        )
        st.caption(
            "Anteil der simulierten Szenarien, in denen der maximale zwischenzeitliche Verlust "
            "des Portfolios mindestens die jeweilige Drawdown-Schwelle erreicht."
        )


        # --------------------------
        # Chart – kombiniert (1 Plot)
        st.subheader("Zeitverlauf – Median & Konfidenzband (Base vs Synth)")
        plot_mc_combined(
            base_band=mc_res.bands["base"],
            syn_band=mc_res.bands["syn"],
            leverage=leverage,
            title=" ",
            y_title="Portfoliowert",
        )

        # --------------------------
        # Histogramm Endwerte
        st.subheader("Histogramm – Endwerte (je Pfad)")
        import plotly.express as px

        COLOR_BASE = "#6e6e6e"   # grau
        COLOR_SYN  = "#1f77b4"   # blau

        end_df = pd.DataFrame({
            "Basis": mc_res.end_values["base"],
            f"Synth {leverage:.1f}x": mc_res.end_values["syn"],
        })

        end_long = end_df.melt(var_name="Produkt", value_name="Endwert")

        fig_hist = px.histogram(
            end_long,
            x="Endwert",
            color="Produkt",
            nbins=90,
            barmode="group",
            title=" ",
            color_discrete_map={
                "Basis": COLOR_BASE,
                f"Synth {leverage:.1f}x": COLOR_SYN,
            },
        )

        # optional: Transparenz für bessere Überlagerung (empfohlen)
        fig_hist.update_traces(opacity=0.6)

        st.plotly_chart(
            fig_hist,
            width="stretch",
            config={"displayModeBar": True},
        )

        # --------------------------
        # Downloads
        st.subheader("Downloads")

        out = pd.DataFrame({
            "end_base": mc_res.path_metrics["end_base"],
            "end_synth": mc_res.path_metrics["end_syn"],
            "total_return_base": mc_res.path_metrics["total_return_base"],
            "total_return_synth": mc_res.path_metrics["total_return_syn"],
            "cagr_base": mc_res.path_metrics["cagr_base"],
            "cagr_synth": mc_res.path_metrics["cagr_syn"],
            "maxdd_base": mc_res.path_metrics["maxdd_base"],
            "maxdd_synth": mc_res.path_metrics["maxdd_syn"],
            "ttr_days_base": mc_res.path_metrics["ttr_days_base"],
            "ttr_days_synth": mc_res.path_metrics["ttr_days_syn"],
            "sharpe_base": mc_res.path_metrics["sharpe_base"],
            "sharpe_synth": mc_res.path_metrics["sharpe_syn"],
        })

        st.download_button(
            "CSV Pfad-Kennzahlen",
            out.to_csv(index=False).encode("utf-8"),
            file_name="mc_path_metrics.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(str(e))
