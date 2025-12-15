from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
import math
import itertools
import random

# ----------------------------
# Helper functions
# ----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def average_pairwise_distance(df):
    pairs = itertools.combinations(df.itertuples(), 2)
    distances = [haversine(p1.lat, p1.lon, p2.lat, p2.lon) for p1, p2 in pairs]
    return sum(distances)/len(distances)

def cluster_area_km2(df):
    lat_km = (df["lat"].max() - df["lat"].min())*111
    lon_km = (df["lon"].max() - df["lon"].min())*111*math.cos(math.radians(df["lat"].mean()))
    return lat_km*lon_km

def add_expected_yield(df):
    df = df.copy()
    df["expected_yield"] = [random.randint(1, 10) for _ in range(len(df))]
    return df

def generate_random_cluster(n_points=5):
    df = pd.DataFrame({
        "city": [f"Point {i+1}" for i in range(n_points)],
        "lat": [random.uniform(59.5, 70.1) for _ in range(n_points)],
        "lon": [random.uniform(20.0, 31.6) for _ in range(n_points)],
    })
    return add_expected_yield(df)

# ----------------------------
# Test DataFrames
# ----------------------------
df1 = pd.DataFrame({
    "city": ["Helsinki", "Tampere", "Turku", "Oulu", "Rovaniemi"],
    "lat": [60.1699, 61.4978, 60.4518, 65.0121, 66.5039],
    "lon": [24.9384, 23.7610, 22.2666, 25.4651, 25.7294],
})

df2 = pd.DataFrame({
    "city": ["Vaasa", "Jyväskylä", "Kuopio", "Kemi", "Kotka"],
    "lat": [63.096, 62.241, 62.892, 65.735, 60.466],
    "lon": [21.615, 25.747, 27.678, 24.567, 26.948],
})

test_dfs = [df1, df2]

# ----------------------------
# Initialize app
# ----------------------------
app = Dash(__name__)
app.layout = html.Div(
    style={"display":"flex", "justifyContent":"center"},
    children=[
        # Sidebar
        html.Div(
            style={"width":"25%", "padding":"20px", "backgroundColor":"#f5f5f5"},
            children=[
                html.H2("Cluster stats"),
                html.Div(id="num-points"),
                html.Div(id="avg-dist"),
                html.Div(id="cluster-area"),
                html.H3("Expected Yield per City"),
                dcc.Graph(id="yield-bar", style={"height":"300px"}),
                html.Button("Next Cluster", id="next-btn", n_clicks=0)
            ]
        ),
        # Map
        html.Div(
            style={"width":"800px"},
            children=[
                html.H1("Finland Dashboard"),
                dcc.Graph(id="map-graph", style={"height":"600px", "width":"800px"})
            ]
        )
    ]
)

# ----------------------------
# Callback to cycle through test dfs
# ----------------------------
@app.callback(
    Output("map-graph", "figure"),
    Output("yield-bar", "figure"),
    Output("num-points", "children"),
    Output("avg-dist", "children"),
    Output("cluster-area", "children"),
    Input("next-btn", "n_clicks")
)
def update_cluster(n_clicks):
    # Generate a new random cluster each click
    df = generate_random_cluster(n_points=5)
    
    # Metrics
    num_points = len(df)
    avg_dist = average_pairwise_distance(df)
    area = cluster_area_km2(df)
    
    # Map
    fig_map = px.scatter_map(
        df,
        lat="lat",
        lon="lon",
        hover_name="city",
        zoom=3,
    )
    fig_map.add_scattermap(
        lat=df["lat"],
        lon=df["lon"],
        mode="lines",
        line=dict(width=2),
        name="Connections"
    )
    
    # Compute center
    lat_center = (df["lat"].max() + df["lat"].min()) / 2
    lon_center = (df["lon"].max() + df["lon"].min()) / 2
    
    # Layout with slight zoom out
    fig_map.update_layout(
        map_style="open-street-map",
        margin={"r":0,"t":0,"l":0,"b":0},
        mapbox_center={"lat": lat_center, "lon": lon_center},
        mapbox_zoom=1 
    )
    
    # Bar chart
    fig_bar = px.bar(
        df,
        x="city",
        y="expected_yield",
        text="expected_yield",
        color="expected_yield",
        color_continuous_scale="Viridis",
        range_y=[0, 10],
        height=300,
    )
    fig_bar.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    
    num_points_text = f"Number of locations: {num_points}"
    avg_dist_text = f"Average distance: {avg_dist:.1f} km"
    area_text = f"Cluster area: {area:,.0f} km²"
    
    return fig_map, fig_bar, num_points_text, avg_dist_text, area_text

# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
