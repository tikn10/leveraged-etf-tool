# ============================
# app.py  (NEU)
# ============================
import streamlit as st
import pandas as pd
import yfinance as yf

from engine import BacktestConfig, TRADING_DAYS_DEFAULT
from backtest import render_backtest_tab
from mc import render_mc_tab

MIN_DATE = pd.to_datetime("1950-01-01")
MAX_DATE = pd.to_datetime("today")

# Zentrale Farben (einheitlich für alle Tabs)
COLOR_MAP = {
    "Basis": "#aec7e8",        # hellblau
    "Synth": "#1f77b4",        # dunkelblau
    "Einzahlungen": "#6e6e6e", # grau
}


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
    Returns (display_name, currency, first_date_available)
    """
    t = (ticker or "").strip()
    if not t:
        return ("", None, None)
    try:
        yt = yf.Ticker(t)
        # Name + Währung
        try:
            info = yt.info or {}
            name = info.get("longName") or info.get("shortName") or t
            currency = info.get("currency")
        except Exception:
            name = t
            currency = None

        # Historie (maximal verfügbar)
        hist = yt.history(period="max", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return (name, currency, None)

        first_dt = pd.to_datetime(hist.index.min()).tz_localize(None).normalize()
        return (name, currency, first_dt)
    except Exception:
        return (t, None, None)



st.set_page_config(page_title="Leveraged ETF Tool", layout="wide")
st.title("Leveraged ETF Tool – Backtesting & Monte Carlo")

st.caption(
    "Dieses Tool dient der simulationsgestützten Analyse von Risiko- und Renditeeigenschaften "
    "gehebelter ETFs im Vergleich zu ungehebelten ETFs. "
    "Die Ergebnisse stellen keine Anlageempfehlung dar, sondern eine methodische "
    "Entscheidungsunterstützung auf Basis historischer Daten und stochastischer Simulationen."
)

st.caption(
    "Methodische Details zur Modellierung, Kennzahlenberechnung "
    "und zu den Limitationen der Simulationen sind der begleitenden Projektarbeit zu entnehmen."
)


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
        help=(
        "Ticker des Basis-ETFs oder Referenzindex. "
       "Es ist der von Yahoo Finance verwendete Ticker zu nutzen "
       "(z. B. URTH (MSCI World), ACWI, SPY (S&P 500), QQQ (Nasdaq 100), ^GSPC)."
        )
    ).strip()

display_name, currency, first_date = get_ticker_meta(base_ticker) if base_ticker else ("", None, None)

if base_ticker:
    parts = [f"Auswahl: **{display_name}** (`{base_ticker}`)"]

    if currency:
        parts.append(f"Währung: **{currency}**")

    if first_date is not None:
        parts.append(f"Frühestes verfügbares Datum: **{first_date.date()}**")
    else:
        parts.append("Frühestes Datum: unbekannt")

    st.caption(" · ".join(parts))


min_start = MIN_DATE
if first_date is not None:
    first_date = pd.to_datetime(first_date).tz_localize(None)
    min_start = max(MIN_DATE, first_date)

default_start = pd.to_datetime("2015-01-01")
start_default_dt = max(default_start, min_start)
start_default_dt = min(start_default_dt, MAX_DATE)

end_default_dt = MAX_DATE
if start_default_dt > end_default_dt:
    start_default_dt = end_default_dt

with c2:
    start_dt = st.date_input(
        "Start Historie",
        value=start_default_dt.date(),
        min_value=min_start.date(),
        max_value=MAX_DATE.date(),
        key="start_date",
        help=(
        "Die maximal verfügbare Historie beginnt ab 1950, "
        "abhängig von der Datenverfügbarkeit des gewählten Basis-Tickers."),
    )

with c3:
    end_dt = st.date_input(
        "Ende Historie",
        value=end_default_dt.date(),
        min_value=min_start.date(),
        max_value=MAX_DATE.date(),
        key="end_date",
    )

start = pd.to_datetime(start_dt).date().isoformat()
end = pd.to_datetime(end_dt).date().isoformat()

# -------------------------------------------------
# Gemeinsamer Block: Investment + Hebel
# -------------------------------------------------
st.markdown("#### Investment")
i1, i2, i3, _ = st.columns([1, 1, 1, 1])

with i1:
    common_start_cap = st.number_input(
        "Startkapital",
        min_value=0,
        value=10_000,
        step=100,
        key="common_start",
    )
with i2:
    common_monthly = st.number_input(
        "Monatliche Sparrate",
        min_value=0,
        value=300,
        step=50,
        key="common_monthly",
    )

with i3:
    common_day = st.number_input(
        "Spar-Tag (1–28)", 
        min_value=1, 
        max_value=28, 
        value=1, 
        step=1, 
        key="common_day",
        help=("Tag des Monats, an dem der Sparplan ausgeführt wird")
        )

st.markdown("#### Synthetischer Hebel")
d1, d2, d3, d4 = st.columns([1, 1, 1, 1])

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
        help="Wird täglich auf den synthetischen ETF umgelegt."
    ) / 100.0

with d3:
    annual_fee = st.number_input(
        "TER p.a. (%)",
        min_value=0.0,
        value=0.5,
        step=0.01,
        key="bt_fee",
        help="Wird täglich auf den synthetischen ETF umgelegt."
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

# Konfiguration, die beide Tabs nutzen
cfg = BacktestConfig(
    start_capital=float(common_start_cap),
    monthly_contribution=float(common_monthly),
    contrib_day=int(common_day),
    slippage_bps=0.0,
    trading_days=int(tdays),
)

st.divider()
tabs = st.tabs(["📈 Backtesting", "🔮 Monte Carlo"])

with tabs[0]:
    render_backtest_tab(
        load_prices=load_prices,
        base_ticker=base_ticker,
        start=start,
        end=end,
        cfg=cfg,
        leverage=float(leverage),
        annual_fin=float(annual_fin),
        annual_fee=float(annual_fee),
        tdays=int(tdays),
        color_map=COLOR_MAP,
    )

with tabs[1]:
    render_mc_tab(
        load_prices=load_prices,
        base_ticker=base_ticker,
        start=start,
        end=end,
        cfg=cfg,
        leverage=float(leverage),
        annual_fin=float(annual_fin),
        annual_fee=float(annual_fee),
        tdays=int(tdays),
        color_map=COLOR_MAP,
    )
