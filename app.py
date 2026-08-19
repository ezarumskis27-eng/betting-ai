import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Betting AI", page_icon="⚽", layout="wide")

DATA = Path("predictions.csv")
COLUMNS = ["date","match","market","odds","probability","fair_odds","edge","ev",
           "confidence","decision","closing_odds","clv","result","profit"]

def load_data():
    if DATA.exists():
        return pd.read_csv(DATA)
    return pd.DataFrame(columns=COLUMNS)

df = load_data()

st.title("⚽ Betting AI")
st.caption("Value • EV • CLV • Performance")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔥 Today's Picks", "📊 Performance", "🧠 Markets", "📜 History"]
)

with tab1:
    st.subheader("Qualified selections")
    bets = df[df["decision"].astype(str).eq("BET")] if not df.empty else df
    if bets.empty:
        st.info("No qualified BETs recorded yet.")
    else:
        st.dataframe(bets.sort_values("ev", ascending=False),
                     use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Model performance")
    if df.empty:
        st.info("Performance appears after predictions are recorded and settled.")
    else:
        profit = pd.to_numeric(df["profit"], errors="coerce").dropna()
        clv = pd.to_numeric(df["clv"], errors="coerce").dropna()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Predictions", len(df))
        c2.metric("Settled", len(profit))
        c3.metric("P/L", f"{profit.sum():.2f}" if len(profit) else "0.00")
        c4.metric("Avg CLV", f"{clv.mean():.2%}" if len(clv) else "—")

with tab3:
    st.subheader("Market monitoring")
    if df.empty:
        st.info("No market history yet.")
    else:
        g = df.groupby("market").agg(
            samples=("market","size"),
            avg_edge=("edge","mean"),
            avg_ev=("ev","mean"),
            avg_clv=("clv","mean")
        ).reset_index()
        g["status"] = g["avg_clv"].apply(
            lambda x: "PROMOTE" if pd.notna(x) and x > 0 else "WATCH"
        )
        st.dataframe(g, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Prediction ledger")
    st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.caption("PASS is a valid result. The system never forces a bet.")
