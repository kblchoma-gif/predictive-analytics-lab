"""3D scatter plot builders (Plotly)."""
import plotly.express as px
from ui.theme import PLOTLY_LAYOUT, THEME


def housing_scatter_3d(df, x="MedInc", y="AveRooms", z="MEDV"):
    sample = df.sample(min(len(df), 1500), random_state=42)
    fig = px.scatter_3d(
        sample, x=x, y=y, z=z, color=z,
        color_continuous_scale="Viridis", opacity=0.65,
        title=f"Housing 3D — {x} × {y} × {z}",
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        height=650,
        scene=dict(
            xaxis_title=x, yaxis_title=y, zaxis_title=z,
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
        coloraxis_colorbar=dict(
            title=z,
            tickfont=dict(color=THEME["primary_text"]),
            title_font=dict(color=THEME["primary_text"]),
        ),
    )
    return fig


def wholesale_clusters_3d(df, labels, x="Grocery", y="Fresh", z="Milk"):
    d = df.copy()
    d["Cluster"] = [f"C{l}" for l in labels]
    fig = px.scatter_3d(
        d, x=x, y=y, z=z, color="Cluster",
        color_discrete_sequence=[
            "#8b0000", "#6a0dad", "#b8860b", "#4b0082",
            "#a52a2a", "#2d1b4e",
        ],
        opacity=0.75,
        title=f"Wholesale Clusters — {x} × {y} × {z}",
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        height=650,
        scene=dict(
            xaxis_title=x, yaxis_title=y, zaxis_title=z,
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