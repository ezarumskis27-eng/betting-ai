import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Betting AI",
    page_icon="⚽",
    layout="wide",
)

DATA = Path("predictions.csv")

COLUMNS = [
    "date", "match", "market", "odds", "probability", "fair_odds",
    "edge", "ev", "confidence", "decision", "closing_odds",
    "clv", "result", "profit"
]


def load_data() -> pd.DataFrame:
    if not DATA.exists():
        return pd.DataFrame(columns=COLUMNS)

    try:
        df = pd.read_csv(DATA)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[COLUMNS]


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


df = load_data()

# Normalize numeric columns.
for col in ["odds", "probability", "fair_odds", "edge", "ev",
            "closing_odds", "clv", "profit"]:
    df[col] = numeric(df, col)

st.title("⚽ Betting AI")
st.caption("Value • EV • CLV • Performance")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔥 Today's Picks", "📊 Performance", "🧠 Markets", "📜 History"]
)

with tab1:
    st.subheader("Qualified selections")

    if df.empty:
        st.info("No predictions recorded yet.")
    else:
        bets = df[df["decision"].astype(str).str.upper().eq("BET")].copy()

        if bets.empty:
            st.info("No qualified BETs found. PASS is a valid result.")
        else:
            bets = bets.sort_values(
                ["ev", "probability"],
                ascending=[False, False],
                na_position="last",
            )

            st.dataframe(
                bets,
                use_container_width=True,
                hide_index=True,
            )

with tab2:
    st.subheader("Model performance")

    if df.empty:
        st.info("Performance appears after predictions are recorded and settled.")
    else:
        profit = df["profit"].dropna()
        clv = df["clv"].dropna()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predictions", len(df))
        c2.metric("Settled", len(profit))
        c3.metric("P/L", f"{profit.sum():.2f}" if not profit.empty else "0.00")
        c4.metric("Avg CLV", f"{clv.mean():.2%}" if not clv.empty else "—")

        if not profit.empty:
            st.subheader("Settled results")
            st.dataframe(
                df[df["profit"].notna()].sort_values("date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

with tab3:
    st.subheader("Market monitoring")

    if df.empty:
        st.info("No market history yet.")
    else:
        work = df.copy()

        g = work.groupby("market", dropna=False).agg(
            samples=("market", "size"),
            avg_edge=("edge", "mean"),
            avg_ev=("ev", "mean"),
            avg_clv=("clv", "mean"),
        ).reset_index()

        g["status"] = g["avg_clv"].apply(
            lambda x: "PROMOTE"
            if pd.notna(x) and x > 0
            else "WATCH"
        )

        st.dataframe(
            g.sort_values("avg_ev", ascending=False, na_position="last"),
            use_container_width=True,
            hide_index=True,
        )

with tab4:
    st.subheader("Prediction ledger")

    if df.empty:
        st.info("Prediction history is empty.")
    else:
        st.dataframe(
            df.sort_values("date", ascending=False, na_position="last"),
            use_container_width=True,
            hide_index=True,
        )

st.divider()
st.caption(
    "PASS is a valid result. The system should never force a bet."
)
