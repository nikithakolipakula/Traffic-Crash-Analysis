# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

# # ======================
# # CONFIG
# # ======================
# st.set_page_config(layout="wide")

# # ======================
# # CACHE (FAST LOAD)
# # ======================
# @st.cache_data
# def load_data():
#     return pd.read_csv("sample_data.csv")

# df = load_data()

# # ======================
# # CLEANING
# # ======================
# for col in df.select_dtypes(include=np.number).columns:
#     df[col].fillna(df[col].mean(), inplace=True)

# for col in df.select_dtypes(include=['object', 'string']).columns:
#     df[col].fillna(df[col].mode()[0], inplace=True)

# df.drop_duplicates(inplace=True)

# # SAMPLE FOR SPEED
# df = df.sample(5000, random_state=42)

# # ======================
# # TITLE
# # ======================
# st.title("🚦 Traffic Intelligence Dashboard")

# # ======================
# # FILTERS
# # ======================
# st.subheader("🎛 Smart Filters")

# c1, c2, c3 = st.columns(3)

# speed_range = c1.slider("Speed Range", 10, 80, (20, 40))
# hour_range = c2.slider("Crash Hour", 0, 23, (0, 23))
# sample_size = c3.slider("Sample Size", 500, 5000, 3000)

# filtered = df[
#     (df["POSTED_SPEED_LIMIT"].between(speed_range[0], speed_range[1])) &
#     (df["CRASH_HOUR"].between(hour_range[0], hour_range[1]))
# ]

# filtered = filtered.sample(min(sample_size, len(filtered)))

# # ======================
# # KPI CARDS
# # ======================
# col1, col2, col3, col4 = st.columns(4)

# col1.metric("Total Records", len(df))
# col2.metric("Filtered", len(filtered))
# col3.metric("Avg Speed", round(filtered["POSTED_SPEED_LIMIT"].mean(), 2))
# col4.metric("Avg Injuries", round(filtered["INJURIES_TOTAL"].mean(), 2))

# st.markdown("---")

# # ======================
# # CHARTS
# # ======================
# row1_col1, row1_col2 = st.columns(2)

# fig1 = px.line(
#     filtered.groupby("CRASH_HOUR").size().reset_index(name="count"),
#     x="CRASH_HOUR",
#     y="count",
#     title="Crash Trend",
#     template="plotly_dark"
# )

# row1_col1.plotly_chart(fig1, use_container_width=True)

# fig2 = px.histogram(
#     filtered,
#     x="POSTED_SPEED_LIMIT",
#     title="Speed Distribution",
#     template="plotly_dark"
# )

# row1_col2.plotly_chart(fig2, use_container_width=True)

# # ======================
# # SECOND ROW
# # ======================
# row2_col1, row2_col2 = st.columns(2)

# fig3 = px.scatter(
#     filtered,
#     x="POSTED_SPEED_LIMIT",
#     y="INJURIES_TOTAL",
#     title="Injuries vs Speed",
#     template="plotly_dark"
# )

# row2_col1.plotly_chart(fig3, use_container_width=True)

# corr = filtered.select_dtypes(include=np.number).iloc[:, :6].corr()

# fig4 = px.imshow(corr, title="Correlation Heatmap", template="plotly_dark")

# row2_col2.plotly_chart(fig4, use_container_width=True)

# # ======================
# # PREDICTION PANEL (FIXED)
# # ======================
# st.markdown("---")
# st.title("🚗 Traffic Crash Prediction")

# # Use only matching features
# features = ["POSTED_SPEED_LIMIT", "LANE_CNT", "CRASH_HOUR"]

# df_model = df[features + ["INJURIES_TOTAL"]].dropna()

# X = df_model[features]
# y = df_model["INJURIES_TOTAL"]

# model = LinearRegression()
# model.fit(X, y)

# # ======================
# # INPUT SLIDERS
# # ======================
# speed = st.slider("Speed", 10, 80, 40)
# lanes = st.slider("Lanes", 1, 6, 2)
# hour = st.slider("Hour", 0, 23, 12)

# # ======================
# # PREDICTION
# # ======================
# if st.button("🚀 Predict"):

#     input_data = np.array([[speed, lanes, hour]])
#     prediction = model.predict(input_data)

#     st.success(f"Predicted Injuries: {round(prediction[0], 2)}")

# # ======================
# # VISUAL (LIKE YOUR IMAGE)
# # ======================
# fig_map = px.scatter_mapbox(
#     filtered,
#     lat="LATITUDE",
#     lon="LONGITUDE",
#     color="INJURIES_TOTAL",
#     size="INJURIES_TOTAL",
#     zoom=10,
#     height=150,   # 👈 reduced size
#     mapbox_style="carto-darkmatter"
# )

# fig_map.update_layout(
#     margin=dict(l=0, r=0, t=20, b=0)
# )

# st.plotly_chart(fig_map, use_container_width=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# ======================
# CONFIG
# ======================
st.set_page_config(layout="wide")

# ======================
# LOAD DATA (FAST)
# ======================
@st.cache_data
def load_data():
    return pd.read_csv("Traffic_Crashes_-_Crashes.csv")

df = load_data()

# ======================
# CLEAN DATA
# ======================
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

df.drop_duplicates(inplace=True)

# SAMPLE FOR PERFORMANCE
df = df.sample(5000, random_state=42)

# ======================
# TITLE
# ======================
st.title("🚦 Traffic Intelligence Dashboard")

# ======================
# FILTERS
# ======================
st.subheader("🎛 Smart Filters")

c1, c2, c3 = st.columns(3)

speed_range = c1.slider("Speed Range", 10, 80, (20, 40))
hour_range = c2.slider("Crash Hour", 0, 23, (0, 23))
sample_size = c3.slider("Sample Size", 500, 5000, 3000)

filtered = df[
    (df["POSTED_SPEED_LIMIT"].between(speed_range[0], speed_range[1])) &
    (df["CRASH_HOUR"].between(hour_range[0], hour_range[1]))
]

filtered = filtered.sample(min(sample_size, len(filtered)))

# ======================
# REMOVE OUTLIERS + CLEAN
# ======================
filtered = filtered.dropna(subset=[
    "POSTED_SPEED_LIMIT",
    "INJURIES_TOTAL",
    "LATITUDE",
    "LONGITUDE"
])

filtered = filtered[filtered["INJURIES_TOTAL"] <= 5]

# ======================
# KPI CARDS
# ======================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(df))
col2.metric("Filtered Records", len(filtered))
col3.metric("Avg Speed", f"{round(filtered['POSTED_SPEED_LIMIT'].mean(),2)} km/h")
col4.metric("Avg Injuries", round(filtered["INJURIES_TOTAL"].mean(), 2))

st.markdown("---")

# ======================
# CHARTS ROW 1
# ======================
row1_col1, row1_col2 = st.columns(2)

fig1 = px.line(
    filtered.groupby("CRASH_HOUR").size().reset_index(name="count"),
    x="CRASH_HOUR",
    y="count",
    title="📈 Crash Trend by Hour",
    template="plotly_dark",
    markers=True
)
fig1.update_traces(line=dict(width=3))
row1_col1.plotly_chart(fig1, use_container_width=True)

fig2 = px.histogram(
    filtered,
    x="POSTED_SPEED_LIMIT",
    nbins=10,
    title="📊 Speed Distribution",
    template="plotly_dark",
    color_discrete_sequence=["#00f5d4"]
)
row1_col2.plotly_chart(fig2, use_container_width=True)

# ======================
# CHARTS ROW 2
# ======================
row2_col1, row2_col2 = st.columns(2)

fig3 = px.scatter(
    filtered,
    x="POSTED_SPEED_LIMIT",
    y="INJURIES_TOTAL",
    title="⚠️ Injuries vs Speed",
    template="plotly_dark",
    color="INJURIES_TOTAL",
    size="INJURIES_TOTAL",
    opacity=0.7
)
row2_col1.plotly_chart(fig3, use_container_width=True)

# CLEAN HEATMAP (MEANINGFUL)
use_cols = [
    "POSTED_SPEED_LIMIT",
    "LANE_CNT",
    "NUM_UNITS",
    "INJURIES_TOTAL"
]

corr = filtered[use_cols].corr()

fig4 = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="teal",
    title="📉 Correlation Heatmap",
    template="plotly_dark"
)
row2_col2.plotly_chart(fig4, use_container_width=True)

# ======================
# MAP
# ======================
st.markdown("---")
st.subheader("📍 Crash Locations")

fig_map = px.scatter_mapbox(
    filtered,
    lat="LATITUDE",
    lon="LONGITUDE",
    color="INJURIES_TOTAL",
    size="INJURIES_TOTAL",
    zoom=10,
    height=250,
    mapbox_style="carto-darkmatter"
)

fig_map.update_layout(margin=dict(l=0, r=0, t=20, b=0))

st.plotly_chart(fig_map, use_container_width=True)

# ======================
# PREDICTION MODEL
# ======================
features = ["POSTED_SPEED_LIMIT", "LANE_CNT", "CRASH_HOUR"]

df_model = df[features + ["INJURIES_TOTAL"]].dropna()

X = df_model[features]
y = df_model["INJURIES_TOTAL"]

model = LinearRegression()
model.fit(X, y)

# ======================
# PREDICTION UI
# ======================
st.markdown("---")
st.subheader("🚗 Live Crash Prediction")

c1, c2, c3 = st.columns(3)

speed = c1.slider("Speed", 10, 80, 40)
lanes = c2.slider("Lanes", 1, 6, 2)
hour = c3.slider("Hour", 0, 23, 12)

prediction = model.predict(np.array([[speed, lanes, hour]]))

st.metric("Predicted Injuries", round(prediction[0], 2))

# ======================
# FINAL SCATTER (ONLY ONE VERSION)
st.subheader("📊 Speed vs Injuries")

fig_scatter = px.scatter(
    df_model,
    x="POSTED_SPEED_LIMIT",
    y="INJURIES_TOTAL",
    title="Speed vs Injuries",
    template="plotly_dark",
    color="INJURIES_TOTAL"
)

st.plotly_chart(fig_scatter, use_container_width=True)
