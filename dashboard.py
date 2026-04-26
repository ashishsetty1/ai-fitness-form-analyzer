import streamlit as st
import pandas as pd
from pathlib import Path

LOG_FILE = Path("workout_log.csv")

st.set_page_config(
    page_title="AI Fitness Form Analyzer Dashboard",
    page_icon="🏋️",
    layout="wide",
)

st.title("🏋️ AI Fitness Form Analyzer Dashboard")
st.write("Track workouts recorded by the computer vision fitness analyzer.")

if not LOG_FILE.exists():
    st.warning("No workout_log.csv found yet.")
    st.info("Run the analyzer first, complete a workout, then press Q to save.")
    st.stop()

df = pd.read_csv(LOG_FILE, on_bad_lines="skip")

if df.empty:
    st.warning("Workout log is empty.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Support both older and newer CSV column names
for col in ["squat_reps", "pushup_reps", "left_curls", "right_curls"]:
    if col not in df.columns:
        df[col] = 0

df["total_curls"] = df["left_curls"] + df["right_curls"]
df["total_reps"] = df["squat_reps"] + df["pushup_reps"] + df["total_curls"]

total_squats = int(df["squat_reps"].sum())
total_pushups = int(df["pushup_reps"].sum())
total_curls = int(df["total_curls"].sum())
total_workouts = len(df)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Workouts", total_workouts)
col2.metric("Total Squats", total_squats)
col3.metric("Total Pushups", total_pushups)
col4.metric("Total Curls", total_curls)

st.divider()

st.subheader("Workout History")
st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

st.subheader("Progress Over Time")

chart_df = df.sort_values("timestamp").set_index("timestamp")[
    ["squat_reps", "pushup_reps", "total_curls", "total_reps"]
]

st.line_chart(chart_df)

st.subheader("Exercise Totals")

totals_df = pd.DataFrame(
    {
        "Exercise": ["Squats", "Pushups", "Curls"],
        "Reps": [total_squats, total_pushups, total_curls],
    }
)

st.bar_chart(totals_df.set_index("Exercise"))

st.divider()

st.subheader("Latest Workout")

latest = df.sort_values("timestamp", ascending=False).iloc[0]

st.write(f"**Date:** {latest['timestamp']}")
st.write(f"**Squats:** {int(latest['squat_reps'])}")
st.write(f"**Pushups:** {int(latest['pushup_reps'])}")
st.write(f"**Left Curls:** {int(latest['left_curls'])}")
st.write(f"**Right Curls:** {int(latest['right_curls'])}")
st.write(f"**Total Reps:** {int(latest['total_reps'])}")