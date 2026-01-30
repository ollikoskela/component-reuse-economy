import json
import math
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

CSV_PATH = "coordinates.csv"

FORSSA = np.array([60.8148, 23.6216])
ROAD_FACTOR = 1.34  # approximate straight-line to road multiplier
GRID_SPACING_KM = 25.0  # spacing for Finland-shaped grid points (map 2)

df = pd.read_csv(CSV_PATH)
points = df[["Latitude", "Longitude"]].to_numpy()

app = Dash(__name__)

def numeric_input(id_, label, value):
    return html.Div([
        html.Label(label),
        dcc.Input(id=id_, type="number", value=value, debounce=True, style={"width": "100%"})
    ])

def haversine_vec(a, b):
    R = 6371
    a = np.radians(a)
    b = np.radians(b)
    dlat = b[:, 0] - a[0]
    dlon = b[:, 1] - a[1]
    x = np.sin(dlat / 2) ** 2 + np.cos(a[0]) * np.cos(b[:, 0]) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(x))

price_adv_grid = np.arange(0, 10001, 500)
cost_km_grid = np.arange(10, 201, 10)


def load_finland_polygons(path="geodata/finland.geojson"):
    with open(path) as f:
        data = json.load(f)
    fin_feat = next(
        (feat for feat in data.get("features", []) if feat.get("properties", {}).get("name") == "Finland"),
        None,
    )
    if fin_feat is None:
        raise ValueError("Finland feature not found in geojson")
    geom = fin_feat["geometry"]
    if geom["type"] == "Polygon":
        polygons = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        polygons = geom["coordinates"]
    else:
        raise ValueError(f"Unsupported geometry type: {geom['type']}")

    min_lon = min(x for poly in polygons for ring in poly for x, y in ring)
    min_lat = min(y for poly in polygons for ring in poly for x, y in ring)
    max_lon = max(x for poly in polygons for ring in poly for x, y in ring)
    max_lat = max(y for poly in polygons for ring in poly for x, y in ring)
    return polygons, (min_lon, min_lat, max_lon, max_lat)


def point_in_ring(x, y, ring):
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1)
        if intersects:
            inside = not inside
    return inside


def point_in_polygon(x, y, polygon):
    outer = polygon[0]
    holes = polygon[1:] if len(polygon) > 1 else []
    if not point_in_ring(x, y, outer):
        return False
    for hole in holes:
        if point_in_ring(x, y, hole):
            return False
    return True


def point_in_multipolygon(x, y, polygons):
    return any(point_in_polygon(x, y, poly) for poly in polygons)


def build_finland_grid(polygons, bounds, spacing_km=GRID_SPACING_KM):
    min_lon, min_lat, max_lon, max_lat = bounds
    lat_step = spacing_km / 111.0  # rough km-to-deg for latitude
    mid_lat = (min_lat + max_lat) / 2.0
    lon_step = spacing_km / (111.0 * math.cos(math.radians(mid_lat)))

    grid_lat, grid_lon = [], []
    lat = min_lat
    while lat <= max_lat:
        lon = min_lon
        while lon <= max_lon:
            if point_in_multipolygon(lon, lat, polygons):
                grid_lat.append(lat)
                grid_lon.append(lon)
            lon += lon_step
        lat += lat_step
    return grid_lat, grid_lon


def build_heat_distance_matrix(polygons, bounds, spacing_km, targets):
    """Compute straight-line distances from each grid point to all targets (adjusted later by ROAD_FACTOR)."""
    heat_lat, heat_lon = build_finland_grid(polygons, bounds, spacing_km)
    matrix = np.zeros((len(heat_lat), len(targets)))

    for idx, (lat_pt, lon_pt) in enumerate(zip(heat_lat, heat_lon)):
        dist_row = haversine_vec(np.array([lat_pt, lon_pt]), targets)
        matrix[idx, :] = dist_row

    return matrix, heat_lat, heat_lon

def build_matrix(distances):
    Z = np.zeros((len(cost_km_grid), len(price_adv_grid)))
    for i, c in enumerate(cost_km_grid):
        for j, p in enumerate(price_adv_grid):
            r = p / c
            effective_r = r / ROAD_FACTOR
            Z[i, j] = np.sum(distances <= effective_r)
    return Z

def make_material_heat(matrix):
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=np.round(price_adv_grid, 1),
            y=np.round(cost_km_grid, 1),
            colorscale="YlGn",
            text=matrix.astype(int),
            texttemplate="%{text}",
            showscale=True
        )
    )
    fig.update_layout(
        title="Material heat – Current location",
        xaxis_title="Price advantage per order (€)",
        yaxis_title="Transport cost per km (€)"
    )
    return fig

dist_points_default = haversine_vec(FORSSA, points)
material_heat = make_material_heat(build_matrix(dist_points_default))
FIN_POLYGONS, FIN_BOUNDS = load_finland_polygons()
HEAT_MATRIX, HEAT_LAT, HEAT_LON = build_heat_distance_matrix(FIN_POLYGONS, FIN_BOUNDS, GRID_SPACING_KM, points)

app.layout = html.Div([
    dcc.Store(id="facility_location", data={"lat": FORSSA[0], "lon": FORSSA[1]}),
    html.H2("2ndHPS Viability – Finland"),

    html.Div([
        html.Div([
            html.H4("Recycling facility"),
            html.Div([
                numeric_input("facility_lat", "Latitude", FORSSA[0]),
                numeric_input("facility_lon", "Longitude", FORSSA[1]),
            ], style={"display": "flex", "gap": "8px"}),
            html.Button("Recalculate location", id="confirm_location", n_clicks=0, style={"margin": "8px 0"}),
            html.Hr(),
            html.H4("Bricks"),
            html.Div([
                numeric_input("brick_price", "Price advantage per order (€)", 2500),
                numeric_input("brick_cost", "Cost per km (€)", 30),
            ], style={"display": "flex", "gap": "8px"}),
            html.Hr(),
            html.H4("Concrete slabs"),
            html.Div([
                numeric_input("slab_price", "Price advantage per order (€)", 5000),
                numeric_input("slab_cost", "Cost per km (€)", 150),
            ], style={"display": "flex", "gap": "8px"}),
            html.Hr(),
            dcc.Graph(id="material_heat", figure=material_heat, style={"height": "45vh"})
        ], style={"width": "30%", "padding": "0 16px", "boxSizing": "border-box"}),

        html.Div([
            html.Div([
                dcc.Graph(id="map_points", style={"height": "100%", "flex": 1}),
            ], style={
                "width": "50%",
                "height": "85vh",
                "border": "1px solid #ccc",
                "borderRadius": "8px",
                "padding": "8px",
                "boxSizing": "border-box",
                "backgroundColor": "#fafafa",
                "display": "flex",
                "flexDirection": "column"
            }),
            html.Div([
                dcc.Graph(id="map_heat", style={"height": "100%", "flex": 1}),
            ], style={
                "width": "50%",
                "height": "85vh",
                "border": "1px solid #ccc",
                "borderRadius": "8px",
                "padding": "8px",
                "boxSizing": "border-box",
                "backgroundColor": "#fafafa",
                "display": "flex",
                "flexDirection": "column"
            }),
        ], style={
            "width": "70%",
            "display": "flex",
            "gap": "12px",
            "justifyContent": "space-between",
            "alignItems": "stretch"
        }),
    ], style={"display": "flex", "gap": "12px"})
])

@app.callback(
    Output("facility_location", "data"),
    Input("confirm_location", "n_clicks"),
    State("facility_lat", "value"),
    State("facility_lon", "value"),
    prevent_initial_call=True
)
def confirm_location(n_clicks, lat, lon):
    return {"lat": lat, "lon": lon}

@app.callback(
    Output("map_points", "figure"),
    Output("map_heat", "figure"),
    Output("material_heat", "figure"),
    Input("brick_price", "value"),
    Input("brick_cost", "value"),
    Input("slab_price", "value"),
    Input("slab_cost", "value"),
    Input("facility_location", "data"),
)
def update_maps(b_price, b_cost, s_price, s_cost, facility):

    center = np.array([facility["lat"], facility["lon"]])
    dist_points = haversine_vec(center, points)

    brick_radius = (b_price / b_cost) / ROAD_FACTOR
    slab_radius = (s_price / s_cost) / ROAD_FACTOR

    brick_access = dist_points <= brick_radius
    slab_access = dist_points <= slab_radius

    fig_points = go.Figure()
    fig_points.add_trace(go.Scattermap(
        lat=points[:, 0],
        lon=points[:, 1],
        mode="markers",
        text=np.select(
            [brick_access & slab_access, brick_access, slab_access],
            ["Brick & slab", "Brick", "Slab"],
            default="Other"
        ),
        marker=dict(
            size=6,
            color=np.select(
                [brick_access & slab_access, brick_access, slab_access],
                ["purple", "green", "blue"],
                default="lightgray"
            ),
            opacity=0.8
        )
    ))
    fig_points.add_trace(go.Scattermap(
        lat=[center[0]],
        lon=[center[1]],
        mode="markers",
        text=["Recycling facility"],
        marker=dict(size=14, color="red", symbol="star")
    ))
    fig_points.update_layout(
        map=dict(style="carto-positron", center={"lat": 64.5, "lon": 26}, zoom=4.8),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False
    )

    heat_val = []
    for row in HEAT_MATRIX:
        heat_val.append(np.sum(row <= brick_radius) + np.sum(row <= slab_radius))

    fig_heat = go.Figure()
    fig_heat.add_trace(go.Scattermap(
        lat=HEAT_LAT,
        lon=HEAT_LON,
        mode="markers",
        text=heat_val,
        marker=dict(size=12, color=heat_val, colorscale="YlOrRd", opacity=0.9, showscale=True, )
    ))
    fig_heat.add_trace(go.Scattermap(
        lat=[center[0]],
        lon=[center[1]],
        mode="markers",
        text=["Recycling facility"],
        marker=dict(size=14, color="red", symbol="star")
    ))
    fig_heat.update_layout(
        map=dict(style="carto-positron", center={"lat": 64.5, "lon": 26}, zoom=4.8),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False
    )

    material_matrix = build_matrix(dist_points)
    material_heat_fig = make_material_heat(material_matrix)

    return fig_points, fig_heat, material_heat_fig

if __name__ == "__main__":
    app.run(debug=True)
