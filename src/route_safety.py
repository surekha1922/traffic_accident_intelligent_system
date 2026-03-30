# clustering.py

from preprocessing import load_and_preprocess
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import HeatMap

# -----------------------------
# STEP 1: Load Data
# -----------------------------
df = load_and_preprocess()

# -----------------------------
# STEP 2: Extract Coordinates
# -----------------------------
coords = df[['Start_Lat', 'Start_Lng']].copy()

# -----------------------------
# STEP 3: Sample Data (for performance)
# -----------------------------
coords = coords.sample(n=10000, random_state=42)

# -----------------------------
# STEP 4: Scale Data (IMPORTANT for DBSCAN)
# -----------------------------
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# -----------------------------
# STEP 5: Apply DBSCAN
# -----------------------------
dbscan = DBSCAN(eps=0.3, min_samples=10)
coords['cluster'] = dbscan.fit_predict(coords_scaled)

# -----------------------------
# STEP 6: Cluster Distribution
# -----------------------------
print("Cluster Distribution:")
print(coords['cluster'].value_counts())

# -----------------------------
# STEP 7: Create Map (Clusters)
# -----------------------------
m = folium.Map(
    location=[coords['Start_Lat'].mean(), coords['Start_Lng'].mean()],
    zoom_start=5
)

# Colors for clusters
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']

# Plot cluster points (skip noise = -1)
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

# Save cluster map
m.save("dbscan_hotspots.html")
print("DBSCAN cluster map saved as dbscan_hotspots.html")

# -----------------------------
# STEP 8: Create Heatmap
# -----------------------------
heat_data = coords[['Start_Lat', 'Start_Lng']].values.tolist()

heatmap = folium.Map(
    location=[coords['Start_Lat'].mean(), coords['Start_Lng'].mean()],
    zoom_start=5
)

HeatMap(
    heat_data,
    radius=8,
    blur=10
).add_to(heatmap)

# Save heatmap
heatmap.save("heatmap.html")
print("Heatmap saved as heatmap.html")