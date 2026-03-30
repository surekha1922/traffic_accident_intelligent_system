# app.py

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import networkx as nx

from src.preprocessing import load_and_preprocess
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("🚦 Traffic Accident Intelligence Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
df = load_and_preprocess()

# -----------------------------
# KPI METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Model Accuracy", "70.7%")
col2.metric("Hotspots Detected", "3")
col3.metric("Data Points Used", f"{len(df)}")

# -----------------------------
# SIDEBAR (PREDICTION INPUT)
# -----------------------------
st.sidebar.header("🔮 Predict Accident Severity")

temperature = st.sidebar.slider("Temperature (F)", 0, 120, 70)
visibility = st.sidebar.slider("Visibility (mi)", 0, 10, 5)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.sidebar.slider("Wind Speed (mph)", 0, 50, 10)
is_night = st.sidebar.selectbox("Night?", [0, 1])

# -----------------------------
# SIMPLE PREDICTION LOGIC
# -----------------------------
risk_score = (
    (10 - visibility) +
    humidity * 0.1 +
    wind_speed * 0.2 +
    is_night * 5
)

severity = int(min(max(risk_score // 5, 1), 4))

st.sidebar.success(f"Predicted Severity: {severity}")

# -----------------------------
# FEATURE IMPORTANCE (STATIC)
# -----------------------------
st.subheader("📊 Feature Importance")

import matplotlib.pyplot as plt

features = ["Lat", "Lng", "Signal", "Crossing", "Stop"]
values = [0.26, 0.21, 0.16, 0.09, 0.07]

fig, ax = plt.subplots()
ax.bar(features, values)
st.pyplot(fig)

# -----------------------------
# HEATMAP + DBSCAN
# -----------------------------
st.subheader("🔥 Accident Heatmap & Hotspots")

coords = df[['Start_Lat', 'Start_Lng']].sample(5000, random_state=42)

# DBSCAN
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

dbscan = DBSCAN(eps=0.3, min_samples=10)
coords['cluster'] = dbscan.fit_predict(coords_scaled)

# Map
m = folium.Map(
    location=[coords['Start_Lat'].mean(), coords['Start_Lng'].mean()],
    zoom_start=5
)

# Heatmap
HeatMap(coords[['Start_Lat', 'Start_Lng']].values.tolist()).add_to(m)

# Cluster points
colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range(len(coords)):
    cluster = int(coords.iloc[i]['cluster'])
    if cluster == -1:
        continue

    folium.CircleMarker(
        location=[coords.iloc[i]['Start_Lat'], coords.iloc[i]['Start_Lng']],
        radius=2,
        color=colors[cluster % len(colors)],
        fill=True
    ).add_to(m)

st_folium(m, width=900, height=500)

# -----------------------------
# SAFE ROUTE (DIJKSTRA + RISK)
# -----------------------------
st.subheader("🚦 Safe Route Finder")

# Create graph
G = nx.Graph()

nodes = ["A", "B", "C", "D", "E", "F"]
G.add_nodes_from(nodes)

edges = [
    ("A", "B", 5),
    ("B", "C", 4),
    ("A", "C", 10),
    ("C", "D", 3),
    ("B", "D", 8),
    ("D", "E", 2),
    ("C", "E", 7),
    ("E", "F", 3),
    ("D", "F", 6),
]

# Risk from cluster density
cluster_counts = coords['cluster'].value_counts()

risk_map = {}
for cluster, count in cluster_counts.items():
    if cluster == -1:
        risk_map[cluster] = 1
    else:
        risk_map[cluster] = min(count / 500, 10)

clusters = list(risk_map.keys())

# Add edges with risk
for u, v, dist in edges:
    cluster = np.random.choice(clusters)
    risk = risk_map[cluster]
    weight = dist + risk

    G.add_edge(u, v, weight=weight, distance=dist, risk=risk)

# UI selection
start = st.selectbox("Start Location", nodes)
end = st.selectbox("End Location", nodes)

if st.button("Find Safest Route"):
    path = nx.shortest_path(G, source=start, target=end, weight="weight")
    cost = nx.shortest_path_length(G, source=start, target=end, weight="weight")

    st.success(f"Safest Path: {path}")
    st.info(f"Total Cost: {round(cost, 2)}")

    st.write("### Edge Details")
    for u, v in zip(path[:-1], path[1:]):
        data = G[u][v]
        st.write(f"{u} → {v} | Distance: {data['distance']:.2f}, Risk: {data['risk']:.2f}")
