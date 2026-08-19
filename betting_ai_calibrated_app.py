import streamlit as st
import pandas as pd
import numpy as np
import math
import requests
from pathlib import Path
from datetime import datetime, timezone

st.set_page_config(page_title="Betting AI", page_icon="⚽", layout="wide")

DATA = Path("predictions.csv")

COLUMNS = [
    "date", "match", "market", "odds", "probability", "fair_odds",
    "edge", "ev", "confidence", "decision", "closing_odds",
    "clv", "result", "profit"
]

st.title("⚽ Betting AI")
st.caption("Probability • Fair Odds • EV • Edge • CLV • Performance")

def poisson_pmf(k, lam):
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_probs(lam_h, lam_a, max_goals=8):
    ph = np.array([poisson_pmf(i, lam_h) for i in range(max_goals + 1)])
    pa = np.array([poisson_pmf(i, lam_a) for i in range(max_goals + 1)])
    matrix = np.outer(ph, pa)
    matrix /= matrix.sum()
    return matrix

def market_probability(matrix, market):
    m = market.strip().lower()
    n = matrix.shape[0] - 1
    if m in ("home win", "1", "home"):
        return float(np.tril(matrix, -0).sum() - np.trace(matrix) - np.triu(matrix, 1).sum()) if False else float(np.tril(matrix, -1).sum())
    if m in ("draw", "x"):
        return float(np.trace(matrix))
    if m in ("away win", "2", "away"):
        return float(np.triu(matrix, 1).sum())
    if m == "btts":
        return float(matrix[1:, 1:].sum())
    if m == "btts no":
        return float(1 - matrix[1:, 1:].sum())
    if m.startswith("over "):
        line = float(m.split()[1])
        return float(sum(matrix[i, j] for i in range(n + 1) for j in range(n + 1) if i + j > line))
    if m.startswith("under "):
        line = float(m.split()[1])
        return float(sum(matrix[i, j] for i in range(n + 1) for j in range(n + 1) if i + j < line))
    if m.startswith("home over "):
        line = float(m.split()[-1])
        return float(sum(matrix[i, :].sum() for i in range(n + 1) if i > line))
    if m.startswith("away over "):
        line = float(m.split()[-1])
        return float(sum(matrix[:, j].sum() for j in range(n + 1) if j > line))
    return np.nan

def load_history():
    if not DATA.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(DATA)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[COLUMNS]

def normalize_history(df):
    for c in ["odds","probability","fair_odds","edge","ev","closing_odds","clv","profit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def poisson_from_history(results):
    req = {"home","away","hg","ag"}
    if not req.issubset(results.columns):
        return None
    r = results.copy()
    for c in ["hg","ag"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.dropna(subset=["home","away","hg","ag"])
    if len(r) < 20:
        return None
    teams = sorted(set(r["home"].astype(str)) | set(r["away"].astype(str)))
    league_h = r["hg"].mean()
    league_a = r["ag"].mean()
    rows = []
    for t in teams:
        home = r[r.home.astype(str) == t]
        away = r[r.away.astype(str) == t]
        hs = home.hg.mean() if len(home) else league_h
        hc = home.ag.mean() if len(home) else league_a
        as_ = away.ag.mean() if len(away) else league_a
        ac = away.hg.mean() if len(away) else league_h
        attack_h = hs / league_h if league_h else 1
        defense_h = hc / league_a if league_a else 1
        attack_a = as_ / league_a if league_a else 1
        defense_a = ac / league_h if league_h else 1
        rows.append([t, attack_h, defense_h, attack_a, defense_a])
    return pd.DataFrame(rows, columns=["team","ah","dh","aa","da"]), league_h, league_a

def model_match(model, league_h, league_a, home, away):
    home = str(home); away = str(away)
    a = model.set_index("team")
    if home not in a.index or away not in a.index:
        return None
    lh = max(0.15, league_h * a.loc[home,"ah"] * a.loc[away,"da"])
    la = max(0.15, league_a * a.loc[away,"aa"] * a.loc[home,"dh"])
    return poisson_probs(lh, la), lh, la

def implied_fair(odds):
    return 1 / odds if odds and odds > 1 else np.nan

def score_bet(p, odds):
    if not np.isfinite(p) or odds <= 1:
        return np.nan, np.nan, np.nan
    fair = 1 / p
    ev = p * odds - 1
    edge = p - (1 / odds)
    return fair, ev, edge

df = normalize_history(load_history())

with st.sidebar:
    st.header("Model input")
    uploaded = st.file_uploader(
        "Upload today's odds CSV",
        type=["csv"],
        help="Columns required: home, away, market, odds. Optional: date."
    )
    hist = st.file_uploader(
        "Upload historical results CSV",
        type=["csv"],
        help="Columns required: home, away, hg, ag. More history = better calibration."
    )
    min_prob = st.slider("Minimum probability", 0.50, 0.90, 0.65, 0.01)
    min_ev = st.slider("Minimum EV", 0.00, 0.20, 0.05, 0.01)
    min_edge = st.slider("Minimum edge", 0.00, 0.20, 0.03, 0.01)

    if st.button("Clear prediction ledger"):
        pd.DataFrame(columns=COLUMNS).to_csv(DATA, index=False)
        st.rerun()

if uploaded is not None and hist is not None:
    try:
        odds_df = pd.read_csv(uploaded)
        hist_df = pd.read_csv(hist)
        fitted = poisson_from_history(hist_df)
        if fitted is None:
            st.error("Historical file needs at least 20 usable matches and columns home, away, hg, ag.")
        else:
            model, league_h, league_a = fitted
            required = {"home","away","market","odds"}
            if not required.issubset(odds_df.columns):
                st.error("Odds file needs columns: home, away, market, odds.")
            else:
                rows = []
                for _, r in odds_df.iterrows():
                    res = model_match(model, league_h, league_a, r["home"], r["away"])
                    if res is None:
                        continue
                    matrix, lh, la = res
                    p = market_probability(matrix, str(r["market"]))
                    odds = float(r["odds"])
                    fair, ev, edge = score_bet(p, odds)
                    # Confidence is deliberately capped below 100 unless the
                    # model has strong probability AND a meaningful market edge.
                    # It is a reporting score, not an extra probability.
                    prob_component = max(0.0, min(1.0, (p - 0.50) / 0.30))
                    edge_component = max(0.0, min(1.0, edge / 0.12))
                    ev_component = max(0.0, min(1.0, ev / 0.15))
                    confidence = round(
                        50 + 20 * prob_component + 15 * edge_component + 15 * ev_component,
                        1
                    )
                    confidence = min(95.0, max(50.0, confidence))
                    decision = "BET" if p >= min_prob and ev >= min_ev and edge >= min_edge else "PASS"
                    rows.append({
                        "date": r.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                        "match": f'{r["home"]} - {r["away"]}',
                        "market": r["market"],
                        "odds": odds,
                        "probability": p,
                        "fair_odds": fair,
                        "edge": edge,
                        "ev": ev,
                        "confidence": confidence,
                        "decision": decision,
                        "closing_odds": np.nan,
                        "clv": np.nan,
                        "result": np.nan,
                        "profit": np.nan,
                    })
                generated = pd.DataFrame(rows, columns=COLUMNS)
                if not generated.empty:
                    generated.to_csv(DATA, index=False)
                    st.success(f"Generated {len(generated)} market predictions.")
                    st.caption(
                        "Confidence is a model-strength score, not a probability. "
                        "It is capped below 100 to avoid false certainty."
                    )
                    st.rerun()
                else:
                    st.warning("No markets could be modeled. Check team names against the historical file.")
    except Exception as e:
        st.error(f"Input error: {e}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔥 Today's Picks", "📊 Performance", "🧠 Markets", "📜 History"]
)

with tab1:
    st.subheader("Qualified selections")
    if df.empty:
        st.info("No predictions recorded yet. Upload odds + historical results in the sidebar.")
    else:
        bets = df[df.decision.astype(str).str.upper().eq("BET")].copy()
        if bets.empty:
            st.info("No qualified BETs found. PASS is a valid result.")
        else:
            bets = bets.sort_values(["ev","probability"], ascending=False, na_position="last")
            show = bets.copy()
            for c in ["probability","edge","ev"]:
                show[c] = show[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
            show["fair_odds"] = show["fair_odds"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            st.dataframe(show, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Model performance")
    if df.empty:
        st.info("Performance appears after predictions are recorded and settled.")
    else:
        profit = df.profit.dropna()
        clv = df.clv.dropna()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Predictions", len(df))
        c2.metric("Settled", len(profit))
        c3.metric("P/L", f"{profit.sum():.2f}" if len(profit) else "0.00")
        c4.metric("Avg CLV", f"{clv.mean():.2%}" if len(clv) else "—")
        if len(profit):
            st.dataframe(df[df.profit.notna()].sort_values("date", ascending=False),
                         use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Market monitoring")
    if df.empty:
        st.info("No market history yet.")
    else:
        g = df.groupby("market", dropna=False).agg(
            samples=("market","size"),
            avg_edge=("edge","mean"),
            avg_ev=("ev","mean"),
            avg_clv=("clv","mean")
        ).reset_index()
        g["status"] = np.where(g.avg_clv > 0, "PROMOTE", "WATCH")
        st.dataframe(g.sort_values("avg_ev", ascending=False, na_position="last"),
                     use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Prediction ledger")
    if df.empty:
        st.info("Prediction history is empty.")
    else:
        st.dataframe(df.sort_values("date", ascending=False, na_position="last"),
                     use_container_width=True, hide_index=True)

st.divider()
st.caption("PASS is a valid result. The model never forces a bet.")
