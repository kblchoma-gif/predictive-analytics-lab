"""
Wednesday-inspired palette.
Dark, restrained, academically serious.
"""
from plotly import graph_objs as go  # noqa: F401


THEME = {
    "background":     "#0a0a0a",
    "surface":        "#141414",
    "surface_alt":    "#1a1a1a",
    "primary_text":   "#e0e0e0",
    "secondary_text": "#888888",
    "accent_crimson": "#8b0000",
    "accent_purple":  "#2d1b4e",
    "border":         "#222222",
    "grid":           "#1a1a1a",
    "success":        "#4a7a3a",
    "warning":        "#a8762c",
    "font":           "Inter, Helvetica, Arial, sans-serif",
}


PLOTLY_LAYOUT = dict(
    paper_bgcolor=THEME["background"],
    plot_bgcolor=THEME["background"],
    font=dict(color=THEME["primary_text"], family=THEME["font"], size=12),
    title=dict(font=dict(size=16, color=THEME["primary_text"])),
    xaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"],
               color=THEME["primary_text"]),
    yaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"],
               color=THEME["primary_text"]),
    margin=dict(l=60, r=30, t=60, b=50),
    legend=dict(bgcolor=THEME["surface"],
                bordercolor=THEME["border"], borderwidth=1),
)