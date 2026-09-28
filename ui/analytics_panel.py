"""Instructor analytics panel — reads from the SQLite event log."""
from dash import dcc, html, dash_table
import plotly.graph_objects as go
from ui.theme import THEME, PLOTLY_LAYOUT
from analytics.logger import fetch_events, session_summary


def build_analytics_panel(session_id: str):
    events = fetch_events(session_id=session_id, limit=200)
    summary = session_summary(session_id)

    timeline_fig = go.Figure()
    if events:
        types = [e[1] for e in events]
        times = [e[0] for e in events]
        timeline_fig.add_trace(go.Scatter(
            x=times, y=list(range(len(times))),
            mode="markers+lines",
            marker=dict(color=THEME["accent_crimson"], size=8),
            text=types,
        ))
    timeline_fig.update_layout(**PLOTLY_LAYOUT)
    timeline_fig.update_layout(
        title="Session Event Timeline",
        xaxis_title="Time", yaxis_title="Event index", height=320,
    )

    table_data = [{"Event Type": k, "Count": v}
                  for k, v in summary.items()] or [
        {"Event Type": "—", "Count": 0}]

    return html.Div([
        html.H3("Learning Analytics", style={
            "fontSize": "13px", "letterSpacing": "2px",
            "color": THEME["accent_crimson"], "marginBottom": "16px",
        }),
        html.Div([
            html.Div([
                html.Div("Session ID", style={
                    "fontSize": "10px", "color": THEME["secondary_text"],
                    "letterSpacing": "1.5px",
                }),
                html.Div(session_id, style={
                    "fontSize": "14px", "fontFamily": "monospace",
                    "color": THEME["primary_text"],
                }),
            ], style={"flex": "1"}),
            html.Div([
                html.Div("Events logged", style={
                    "fontSize": "10px", "color": THEME["secondary_text"],
                    "letterSpacing": "1.5px",
                }),
                html.Div(str(sum(summary.values())), style={
                    "fontSize": "22px", "fontWeight": "600",
                    "color": THEME["accent_crimson"],
                }),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "gap": "32px",
                   "marginBottom": "20px"}),
        dcc.Graph(figure=timeline_fig,
                  config={"displayModeBar": False}),
        dash_table.DataTable(
            data=table_data,
            columns=[{"name": c, "id": c} for c in ["Event Type", "Count"]],
            style_header={
                "backgroundColor": THEME["surface_alt"],
                "color": THEME["primary_text"],
                "fontWeight": "600", "fontSize": "11px",
            },
            style_cell={
                "backgroundColor": THEME["background"],
                "color": THEME["primary_text"],
                "fontSize": "12px", "textAlign": "left",
                "padding": "6px 12px",
            },
            page_size=10,
        ),
    ], style={
        "backgroundColor": THEME["surface"],
        "border": f"1px solid {THEME['border']}",
        "borderRadius": "4px", "padding": "20px",
    })