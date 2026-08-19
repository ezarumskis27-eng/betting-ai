import math
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st


# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="Betting AI",
    page_icon="⚽",
    layout="wide",
)

BASE_URL = "https://v3.football.api-sports.io"

COLUMNS = [
    "date",
    "match",
    "market",
    "odds",
    "probability",
    "fair_odds",
    "edge",
    "ev",
    "confidence",
    "decision",
    "closing_odds",
    "clv",
    "result",
    "profit",
]


# =========================
# API
# =========================

def get_api_key():
    try:
        return st.secrets["API_FOOTBALL_KEY"]
    except Exception:
        return None


def api_get(endpoint, params=None):
    key = get_api_key()

    if not key:
        return None, "API_FOOTBALL_KEY nerastas Secrets."

    headers = {
        "x-apisports-key": key,
        "Accept": "application/json",
    }

    try:
        r = requests.get(
            BASE_URL + endpoint,
            headers=headers,
            params=params or {},
            timeout=20,
        )

        if r.status_code != 200:
            return None, f"API HTTP klaida {r.status_code}: {r.text[:300]}"

        data = r.json()

        if data.get("errors"):
            return None, f"API klaida: {data['errors']}"

        return data.get("response", []), None

    except Exception as e:
        return None, f"Prisijungimo klaida: {e}"


# =========================
# HELPERS
# =========================

def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0

    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def probability_over_25(home_goals, away_goals):
    total = 0.0

    for h in range(0, 8):
        for a in range(0, 8):
            if h + a >= 3:
                total += poisson_pmf(h, home_goals) * poisson_pmf(a, away_goals)

    return min(max(total, 0.0), 1.0)


def probability_under_25(home_goals, away_goals):
    return 1.0 - probability_over_25(home_goals, away_goals)


def probability_btts(home_goals, away_goals):
    ph = 1 - poisson_pmf(0, home_goals)
    pa = 1 - poisson_pmf(0, away_goals)
    return ph * pa


def normalize_probability(value):
    try:
        return float(value) / 100.0
    except Exception:
        return None


def fair_odds(prob):
    if prob is None or prob <= 0:
        return None
    return 1.0 / prob


def calculate_edge(prob, odds):
    if prob is None or odds is None:
        return None
    return prob * odds - 1.0


def confidence_from_edge(edge, probability):
    if edge is None or probability is None:
        return 0

    score = (
        probability * 70
        + max(edge, 0) * 100
    )

    return min(99, max(1, score))


# =========================
# ODDS PARSER
# =========================

def get_best_market_odds(fixture_id):
    response, error = api_get(
        "/odds",
        {"fixture": fixture_id}
    )

    if error or not response:
        return []

    results = []

    for bookmaker in response:
        bookmaker_name = bookmaker.get("bookmaker", {}).get(
            "name", "Unknown"
        )

        for bet in bookmaker.get("bets", []):
            bet_name = str(bet.get("name", "")).lower()

            for value in bet.get("values", []):
                label = str(value.get("value", ""))
                odd_raw = value.get("odd")

                try:
                    odd = float(odd_raw)
                except Exception:
                    continue

                if odd <= 1.01:
                    continue

                results.append({
                    "bookmaker": bookmaker_name,
                    "market": bet_name,
                    "label": label,
                    "odds": odd,
                })

    return results


def find_odds(odds_data, market_type):
    candidates = []

    for item in odds_data:
        market = item["market"]
        label = item["label"]

        if market_type == "home":
            if "match winner" in market and label.lower() in ["home", "1"]:
                candidates.append(item)

        elif market_type == "draw":
            if "match winner" in market and label.lower() in ["draw", "x"]:
                candidates.append(item)

        elif market_type == "away":
            if "match winner" in market and label.lower() in ["away", "2"]:
                candidates.append(item)

        elif market_type == "over25":
            if "over/under" in market and label.lower() in ["over 2.5", "over"]:
                candidates.append(item)

        elif market_type == "under25":
            if "over/under" in market and label.lower() in ["under 2.5", "under"]:
                candidates.append(item)

        elif market_type == "btts_yes":
            if "both teams to score" in market and label.lower() in ["yes"]:
                candidates.append(item)

        elif market_type == "btts_no":
            if "both teams to score" in market and label.lower() in ["no"]:
                candidates.append(item)

    if not candidates:
        return None

    return max(candidates, key=lambda x: x["odds"])


# =========================
# PREDICTION
# =========================

def get_prediction(fixture_id):
    response, error = api_get(
        "/predictions",
        {"fixture": fixture_id}
    )

    if error or not response:
        return None

    try:
        item = response[0]
        prediction = item.get("predictions", {})

        percent = prediction.get("percent", {})

        home_prob = normalize_probability(
            percent.get("home")
        )
        draw_prob = normalize_probability(
            percent.get("draw")
        )
        away_prob = normalize_probability(
            percent.get("away")
        )

        goals = prediction.get("goals", {})

        home_goals = float(
            goals.get("home") or 1.0
        )

        away_goals = float(
            goals.get("away") or 1.0
        )

        return {
            "home_prob": home_prob,
            "draw_prob": draw_prob,
            "away_prob": away_prob,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "advice": prediction.get("advice", ""),
            "under_over": prediction.get("under_over", ""),
        }

    except Exception:
        return None


# =========================
# BUILD BETS
# =========================

def analyse_fixture(fixture):
    fixture_id = fixture["fixture"]["id"]

    home = fixture["teams"]["home"]["name"]
    away = fixture["teams"]["away"]["name"]

    kickoff = fixture["fixture"]["date"]

    prediction = get_prediction(fixture_id)

    if not prediction:
        return []

    odds_data = get_best_market_odds(fixture_id)

    bets = []

    markets = [
        (
            "Home Win",
            prediction["home_prob"],
            find_odds(odds_data, "home"),
        ),
        (
            "Draw",
            prediction["draw_prob"],
            find_odds(odds_data, "draw"),
        ),
        (
            "Away Win",
            prediction["away_prob"],
            find_odds(odds_data, "away"),
        ),
    ]

    over_prob = probability_over_25(
        prediction["home_goals"],
        prediction["away_goals"],
    )

    under_prob = 1.0 - over_prob

    btts_prob = probability_btts(
        prediction["home_goals"],
        prediction["away_goals"],
    )

    markets += [
        (
            "Over 2.5",
            over_prob,
            find_odds(odds_data, "over25"),
        ),
        (
            "Under 2.5",
            under_prob,
            find_odds(odds_data, "under25"),
        ),
        (
            "BTTS Yes",
            btts_prob,
            find_odds(odds_data, "btts_yes"),
        ),
        (
            "BTTS No",
            1.0 - btts_prob,
            find_odds(odds_data, "btts_no"),
        ),
    ]

    for market, probability, odd_data in markets:

        if probability is None or odd_data is None:
            continue

        odds = odd_data["odds"]

        edge = calculate_edge(
            probability,
            odds
        )

        fair = fair_odds(probability)

        if edge is None or fair is None:
            continue

        confidence = confidence_from_edge(
            edge,
            probability
        )

        # Strict qualification.
        # We don't force bets.
        if probability < 0.60:
            continue

        if edge < 0.05:
            continue

        decision = "BET"

        bets.append({
            "date": kickoff,
            "match": f"{home} - {away}",
            "market": market,
            "odds": round(odds, 2),
            "probability": round(probability * 100, 1),
            "fair_odds": round(fair, 2),
            "edge": round(edge * 100, 2),
            "ev": round(edge * 100, 2),
            "confidence": round(confidence, 1),
            "decision": decision,
            "closing_odds": "",
            "clv": "",
            "result": "",
            "profit": "",
        })

    return bets


# =========================
# LOAD TODAY
# =========================

@st.cache_data(ttl=300)
def load_today():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    fixtures, error = api_get(
        "/fixtures",
        {
            "date": today,
            "timezone": "Europe/Oslo",
        },
    )

    if error:
        return pd.DataFrame(columns=COLUMNS), error

    if not fixtures:
        return pd.DataFrame(columns=COLUMNS), None

    all_bets = []

    # Limit the number of fixtures so we don't
    # burn the API quota unnecessarily.
    for fixture in fixtures[:30]:

        status = fixture["fixture"]["status"]["short"]

        if status not in ["NS", "TBD"]:
            continue

        bets = analyse_fixture(fixture)

        all_bets.extend(bets)

    if not all_bets:
        return pd.DataFrame(columns=COLUMNS), None

    df = pd.DataFrame(all_bets)

    df = df.sort_values(
        ["ev", "confidence"],
        ascending=False
    )

    return df, None


# =========================
# SAVE
# =========================

def save_predictions(df):
    if df.empty:
        return

    try:
        df.to_csv(
            "predictions.csv",
            index=False
        )
    except Exception:
        pass


# =========================
# UI
# =========================

st.title("⚽ Betting AI")
st.caption("Value • EV • CLV • Performance")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔥 Today's Picks",
        "📊 Performance",
        "🧠 Markets",
        "📜 History",
    ]
)

with st.spinner("Analysing today's football markets..."):

    df, error = load_today()

if error:
    st.error(error)

with tab1:

    st.subheader("Qualified selections")

    if df.empty:

        st.info(
            "No qualified BETs found. "
            "The system does not force bets."
        )

    else:

        display = df[
            [
                "match",
                "market",
                "odds",
                "probability",
                "fair_odds",
                "edge",
                "ev",
                "confidence",
                "decision",
            ]
        ].copy()

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        st.success(
            f"{len(display)} qualified BET(s) found."
        )


with tab2:

    st.subheader("Model performance")

    if df.empty:

        st.info("No predictions yet.")

    else:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Predictions",
            len(df)
        )

        c2.metric(
            "Qualified BETs",
            int((df["decision"] == "BET").sum())
        )

        c3.metric(
            "Average EV",
            f"{df['ev'].mean():.2f}%"
        )

        c4.metric(
            "Average CLV",
            "-"
        )

        st.dataframe(
            df[
                [
                    "match",
                    "market",
                    "odds",
                    "probability",
                    "ev",
                    "confidence",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


with tab3:

    st.subheader("Market monitoring")

    if df.empty:

        st.info("No market history yet.")

    else:

        g = (
            df.groupby("market")
            .agg(
                samples=("market", "size"),
                avg_edge=("edge", "mean"),
                avg_ev=("ev", "mean"),
                avg_confidence=("confidence", "mean"),
            )
            .reset_index()
        )

        g["status"] = g["avg_ev"].apply(
            lambda x:
            "PROMOTE" if x > 0 else "WATCH"
        )

        st.dataframe(
            g,
            use_container_width=True,
            hide_index=True,
        )


with tab4:

    st.subheader("Prediction ledger")

    if df.empty:

        st.info("No prediction history yet.")

    else:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


st.divider()

st.caption(
    "PASS is a valid result. "
    "The system never forces a bet."
)

# Save current qualified selections
save_predictions(df)
