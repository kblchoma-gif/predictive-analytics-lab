"""3D prediction surface builder (Plotly)."""
import numpy as np
import plotly.graph_objects as go
from models.predict import predict_grid
from ui.theme import PLOTLY_LAYOUT, THEME


def housing_surface_3d(model: str = None, n_points: int = 25):
    medinc_vals = np.linspace(1.0, 10.0, n_points)
    aveoccup_vals = np.linspace(1.0, 6.0, n_points)
    Z = predict_grid(medinc_vals, aveoccup_vals, model=model)

    fig = go.Figure(data=[go.Surface(
        x=medinc_vals, y=aveoccup_vals, z=Z,
        colorscale="Viridis",
        colorbar=dict(
            title="Predicted MEDV",
            tickfont=dict(color=THEME["primary_text"]),
            title_font=dict(color=THEME["primary_text"]),
        ),
    )])
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        height=650,
        title="Prediction Surface — MedInc × AveOccup",
        scene=dict(
            xaxis_title="MedInc",
            yaxis_title="AveOccup",
            zaxis_title="Predicted MEDV ($100k)",
            bgcolor=THEME["background"],
            xaxis=dict(gridcolor=THEME["grid"],
                       backgroundcolor=THEME["background"],
                       color=THEME["primary_text"]),
            yaxis=dict(gridcolor=THEME["grid"],
                       backgroundcolor=THEME["background"],
                       color=THEME["primary_text"]),
            zaxis=dict(gridcolor=THEME["grid"],
                       backgroundcolor=THEME["background"],
                       color=THEME["primary_text"]),
        ),
    )
    return fig