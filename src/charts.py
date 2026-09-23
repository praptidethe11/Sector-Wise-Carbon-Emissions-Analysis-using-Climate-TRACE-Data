import pandas as pd
import plotly.graph_objects as go
from src.state_coords import STATE_COORDS


def sector_shap_series(sector_shap: pd.DataFrame) -> pd.Series:
    """Return a sector -> mean SHAP value series, ready for st.bar_chart."""
    return sector_shap.set_index("sector")["shap_value"]


def top_states_shap_series(state_shap: pd.DataFrame, n: int = 10) -> pd.Series:
    """Return the top-n states by mean SHAP value, ready for st.bar_chart."""
    top = state_shap.nlargest(n, "shap_value")
    return top.set_index("state")["shap_value"]

def build_gas_map(map_data: pd.DataFrame, sector: str, gases: list, year: int):
    filtered = map_data[(map_data["sector"] == sector) & (map_data["year"] == year)]
    state_totals = (
        filtered[filtered["gas"].isin(gases)]
        .groupby("state")["value"].sum().reset_index()
    )
    state_totals["lat"] = state_totals["state"].map(lambda s: STATE_COORDS.get(s, (None, None))[0])
    state_totals["lon"] = state_totals["state"].map(lambda s: STATE_COORDS.get(s, (None, None))[1])
    state_totals = state_totals.dropna(subset=["lat", "lon"])

    max_val = state_totals["value"].max()
    sizes = 10 + (state_totals["value"] / max_val) * 45 if max_val > 0 else 10

    fig = go.Figure(go.Scattermap(
        lat=state_totals["lat"], lon=state_totals["lon"],
        mode="markers",
        marker=dict(size=sizes, color="#5DCAA5", opacity=0.8),
        customdata=state_totals[["state", "value"]],
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,.0f} t<extra></extra>",
    ))
    fig.update_layout(
        map_style="carto-darkmatter",
        map_zoom=3.6,
        map_center={"lat": 22.5, "lon": 80},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=560,
        paper_bgcolor="#1a1a1a",
        font_color="white",
    )
    return fig, state_totals