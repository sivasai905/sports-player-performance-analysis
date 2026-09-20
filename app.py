
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="IPL Player Performance Analysis",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 IPL Cricket Player Performance Analysis")
st.caption("Interactive IPL ball-by-ball data analytics dashboard")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
DATA_PATH = "/content/ipl_deliveries.zip"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, compression="zip")
    return df

    required_columns = [
        "match_id", "date", "venue", "innings",
        "batting_team", "over", "ball", "batter",
        "bowler", "batter_runs", "total_runs",
        "extras", "wickets"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in [
        "innings", "over", "ball", "batter_runs",
        "total_runs", "extras", "wickets"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["batter"] = df["batter"].fillna("Unknown")
    df["bowler"] = df["bowler"].fillna("Unknown")
    df["venue"] = df["venue"].fillna("Unknown")
    df["batting_team"] = df["batting_team"].fillna("Unknown")

    return df


if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Upload ipl_deliveries.csv to your Colab session.")
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("🔎 Dashboard Filters")

teams = sorted(df["batting_team"].dropna().unique())
selected_teams = st.sidebar.multiselect(
    "Select batting teams",
    teams,
    default=teams
)

players = sorted(df["batter"].dropna().unique())
selected_players = st.sidebar.multiselect(
    "Select batters",
    players
)

venues = sorted(df["venue"].dropna().unique())
selected_venues = st.sidebar.multiselect(
    "Select venues",
    venues,
    default=venues
)

innings_options = sorted(df["innings"].unique())
selected_innings = st.sidebar.multiselect(
    "Select innings",
    innings_options,
    default=innings_options
)

filtered = df[
    df["batting_team"].isin(selected_teams)
    & df["venue"].isin(selected_venues)
    & df["innings"].isin(selected_innings)
].copy()

if selected_players:
    filtered = filtered[
        filtered["batter"].isin(selected_players)
    ]

if filtered.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def batter_stats(data):
    result = data.groupby("batter").agg(
        Runs=("batter_runs", "sum"),
        Balls=("batter_runs", "count"),
        Matches=("match_id", "nunique"),
        Fours=("batter_runs", lambda x: (x == 4).sum()),
        Sixes=("batter_runs", lambda x: (x == 6).sum())
    ).reset_index()

    result["Strike Rate"] = np.where(
        result["Balls"] > 0,
        result["Runs"] / result["Balls"] * 100,
        0
    )

    result["Boundary Runs"] = (
        result["Fours"] * 4 + result["Sixes"] * 6
    )

    return result.sort_values("Runs", ascending=False)


def bowler_stats(data):
    result = data.groupby("bowler").agg(
        Runs_Conceded=("total_runs", "sum"),
        Deliveries=("total_runs", "count"),
        Wickets=("wickets", "sum"),
        Matches=("match_id", "nunique")
    ).reset_index()

    result["Overs"] = result["Deliveries"] / 6

    result["Economy"] = np.where(
        result["Overs"] > 0,
        result["Runs_Conceded"] / result["Overs"],
        0
    )

    result["Bowling Average"] = np.where(
        result["Wickets"] > 0,
        result["Runs_Conceded"] / result["Wickets"],
        np.nan
    )

    return result.sort_values("Wickets", ascending=False)


bat_stats = batter_stats(filtered)
bowl_stats = bowler_stats(filtered)

# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------
tabs = st.tabs([
    "📊 Overview",
    "🏏 Batter Analysis",
    "🎯 Bowler Analysis",
    "🛡️ Team Analysis",
    "🏟️ Venue Analysis",
    "📁 Dataset Explorer"
])

# ==================================================
# TAB 1: OVERVIEW
# ==================================================
with tabs[0]:

    st.subheader("Dashboard Overview")

    total_runs = int(filtered["total_runs"].sum())
    total_matches = filtered["match_id"].nunique()
    total_batters = filtered["batter"].nunique()
    total_bowlers = filtered["bowler"].nunique()
    total_wickets = int(filtered["wickets"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Matches", f"{total_matches:,}")
    c2.metric("Total Runs", f"{total_runs:,}")
    c3.metric("Wickets", f"{total_wickets:,}")
    c4.metric("Batters", f"{total_batters:,}")
    c5.metric("Bowlers", f"{total_bowlers:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("### Top 10 Run Scorers")

        top_batters = bat_stats.head(10)

        fig = px.bar(
            top_batters.sort_values("Runs"),
            x="Runs",
            y="batter",
            orientation="h",
            title="Top Batters by Runs",
            labels={"batter": "Batter"}
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Top 10 Wicket Takers")

        top_bowlers = bowl_stats.head(10)

        fig = px.bar(
            top_bowlers.sort_values("Wickets"),
            x="Wickets",
            y="bowler",
            orientation="h",
            title="Top Bowlers by Wickets",
            labels={"bowler": "Bowler"}
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Runs by Batting Team")

    team_runs = (
        filtered.groupby("batting_team")["total_runs"]
        .sum()
        .reset_index()
        .sort_values("total_runs", ascending=False)
    )

    fig = px.bar(
        team_runs,
        x="batting_team",
        y="total_runs",
        color="total_runs",
        title="Total Runs by Team"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset Summary")

    st.write({
        "Rows": len(filtered),
        "Columns": len(filtered.columns),
        "Date Range": (
            f"{filtered['date'].min().date()} to "
            f"{filtered['date'].max().date()}"
            if filtered["date"].notna().any()
            else "Not available"
        )
    })


# ==================================================
# TAB 2: BATTER ANALYSIS
# ==================================================
with tabs[1]:

    st.subheader("🏏 Batter Performance Analysis")

    batter_list = sorted(filtered["batter"].unique())

    selected_batter = st.selectbox(
        "Choose a batter",
        batter_list,
        key="batter_select"
    )

    player_df = filtered[
        filtered["batter"] == selected_batter
    ].copy()

    runs = int(player_df["batter_runs"].sum())
    balls = len(player_df)
    matches = player_df["match_id"].nunique()

    fours = int((player_df["batter_runs"] == 4).sum())
    sixes = int((player_df["batter_runs"] == 6).sum())

    strike_rate = runs / balls * 100 if balls else 0

    a, b, c, d, e = st.columns(5)

    a.metric("Runs", f"{runs:,}")
    b.metric("Balls Faced", f"{balls:,}")
    c.metric("Strike Rate", f"{strike_rate:.2f}")
    d.metric("Fours", fours)
    e.metric("Sixes", sixes)

    st.markdown("### Runs by Match")

    match_runs = (
        player_df.groupby("match_id")["batter_runs"]
        .sum()
        .reset_index()
    )

    match_runs["Match"] = range(1, len(match_runs) + 1)

    fig = px.line(
        match_runs,
        x="Match",
        y="batter_runs",
        markers=True,
        title=f"{selected_batter} - Runs by Match"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Runs Distribution")

    run_distribution = (
        player_df.groupby("batter_runs")
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        run_distribution,
        x="batter_runs",
        y="Count",
        title="Scoring Distribution per Delivery"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Batter Statistics Table")

    st.dataframe(
        bat_stats,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# TAB 3: BOWLER ANALYSIS
# ==================================================
with tabs[2]:

    st.subheader("🎯 Bowler Performance Analysis")

    bowler_list = sorted(filtered["bowler"].unique())

    selected_bowler = st.selectbox(
        "Choose a bowler",
        bowler_list,
        key="bowler_select"
    )

    bowler_df = filtered[
        filtered["bowler"] == selected_bowler
    ].copy()

    conceded = int(bowler_df["total_runs"].sum())
    deliveries = len(bowler_df)
    wickets = int(bowler_df["wickets"].sum())
    overs = deliveries / 6
    economy = conceded / overs if overs else 0

    a, b, c, d = st.columns(4)

    a.metric("Runs Conceded", conceded)
    b.metric("Wickets", wickets)
    c.metric("Deliveries", deliveries)
    d.metric("Economy (approx.)", f"{economy:.2f}")

    st.markdown("### Wickets by Bowler")

    fig = px.bar(
        bowl_stats.head(15).sort_values("Wickets"),
        x="Wickets",
        y="bowler",
        orientation="h",
        title="Top 15 Bowlers by Wickets"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Runs Conceded by Bowler")

    fig = px.scatter(
        bowl_stats,
        x="Runs_Conceded",
        y="Wickets",
        hover_name="bowler",
        size="Deliveries",
        title="Bowler Runs vs Wickets"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Bowler Statistics Table")

    st.dataframe(
        bowl_stats,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Note: This dataset's wickets field counts dismissal events "
        "in each delivery. It does not distinguish wickets credited "
        "to the bowler. Economy is approximate because the dataset "
        "does not identify legal deliveries or wides/no-balls separately."
    )


# ==================================================
# TAB 4: TEAM ANALYSIS
# ==================================================
with tabs[3]:

    st.subheader("🛡️ Team Performance Analysis")

    team_stats = filtered.groupby("batting_team").agg(
        Total_Runs=("total_runs", "sum"),
        Batter_Runs=("batter_runs", "sum"),
        Matches=("match_id", "nunique"),
        Wickets_Lost=("wickets", "sum"),
        Deliveries=("total_runs", "count")
    ).reset_index()

    team_stats["Run Rate (approx.)"] = np.where(
        team_stats["Deliveries"] > 0,
        team_stats["Total_Runs"] /
        (team_stats["Deliveries"] / 6),
        0
    )

    st.markdown("### Team Runs")

    fig = px.bar(
        team_stats.sort_values("Total_Runs", ascending=False),
        x="batting_team",
        y="Total_Runs",
        color="Total_Runs",
        title="Total Runs by Team"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Team Wickets")

    fig = px.bar(
        team_stats.sort_values("Wickets_Lost", ascending=False),
        x="batting_team",
        y="Wickets_Lost",
        title="Dismissal Events by Batting Team"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Team Statistics")

    st.dataframe(
        team_stats,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# TAB 5: VENUE ANALYSIS
# ==================================================
with tabs[4]:

    st.subheader("🏟️ Venue Performance Analysis")

    venue_stats = filtered.groupby("venue").agg(
        Total_Runs=("total_runs", "sum"),
        Matches=("match_id", "nunique"),
        Wickets=("wickets", "sum"),
        Deliveries=("total_runs", "count")
    ).reset_index()

    venue_stats["Runs per Delivery"] = np.where(
        venue_stats["Deliveries"] > 0,
        venue_stats["Total_Runs"] /
        venue_stats["Deliveries"],
        0
    )

    st.markdown("### Total Runs by Venue")

    fig = px.bar(
        venue_stats.sort_values("Total_Runs", ascending=False),
        x="venue",
        y="Total_Runs",
        title="Runs by Venue"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Average Runs per Delivery by Venue")

    fig = px.bar(
        venue_stats.sort_values(
            "Runs per Delivery",
            ascending=False
        ),
        x="venue",
        y="Runs per Delivery",
        title="Runs per Delivery by Venue"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Venue Statistics")

    st.dataframe(
        venue_stats,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# TAB 6: DATASET EXPLORER
# ==================================================
with tabs[5]:

    st.subheader("📁 Dataset Explorer")

    st.write(f"Filtered records: {len(filtered):,}")

    search_text = st.text_input(
        "Search batter, bowler, team, or venue"
    )

    display_df = filtered.copy()

    if search_text:
        mask = (
            display_df["batter"].astype(str).str.contains(
                search_text, case=False, na=False
            )
            | display_df["bowler"].astype(str).str.contains(
                search_text, case=False, na=False
            )
            | display_df["batting_team"].astype(str).str.contains(
                search_text, case=False, na=False
            )
            | display_df["venue"].astype(str).str.contains(
                search_text, case=False, na=False
            )
        )

        display_df = display_df[mask]

    rows_per_page = st.selectbox(
        "Rows to display",
        [10, 25, 50, 100, 500],
        index=1
    )

    st.dataframe(
        display_df.head(rows_per_page),
        use_container_width=True,
        hide_index=True
    )

    csv_data = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download filtered CSV",
        data=csv_data,
        file_name="filtered_ipl_deliveries.csv",
        mime="text/csv"
    )

    st.markdown("### Missing Values")

    missing_df = (
        filtered.isnull()
        .sum()
        .reset_index()
    )

    missing_df.columns = ["Column", "Missing Values"]

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Dataset Information")

    st.write({
        "Rows": len(filtered),
        "Columns": len(filtered.columns),
        "Memory Usage (MB)": round(
            filtered.memory_usage(deep=True).sum()
            / (1024 ** 2),
            2
        )
    })

st.divider()
st.caption("IPL Cricket Player Performance Analysis | Built with Streamlit")
