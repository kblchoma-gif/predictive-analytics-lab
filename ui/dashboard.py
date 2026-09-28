"""
Main Dash application.

Ten panels:
  DATASET | VISUALIZATION | ML | EXPLAINABILITY | UNCERTAINTY
  | PREDICTION | IMMERSIVE | CASE STUDIES | ASSESSMENT | ANALYTICS

Consumes the advanced model results structure from
`models.advanced_train.evaluate_all_models`.

Flask routes /immersive and /data/<file> serve the WebXR viewer and
its point-cloud JSON. These bypass the Dash assets whitelist, which
does not reliably serve arbitrary .html files.
"""
import os
import uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from flask import send_from_directory

from ui.theme import THEME, PLOTLY_LAYOUT
from ui.analytics_panel import build_analytics_panel
from visualization.scatter3d import housing_scatter_3d, wholesale_clusters_3d
from visualization.surface3d import housing_surface_3d
from models.predict import predict_housing, _best_model
from models.uncertainty import conformal_intervals
from models.counterfactual import counterfactual_regression
from analytics.logger import log_event
from learning.bkt import AdaptiveAssessment
from learning.question_bank import QUESTION_BANK


HOUSING_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveOccup"]
FEATURE_BOUNDS = [
    (0.5, 15.0),   # MedInc
    (1.0, 52.0),   # HouseAge
    (2.0, 10.0),   # AveRooms
    (1.0, 8.0),    # AveOccup
]


# ---------------------------------------------------------------------------
# Small reusable components
# ---------------------------------------------------------------------------

def _card(title, children, style=None):
    base = {
        "backgroundColor": THEME["surface"],
        "border": f"1px solid {THEME['border']}",
        "borderRadius": "4px",
        "padding": "18px 20px",
        "marginBottom": "16px",
        "color": THEME["primary_text"],
    }
    if style:
        base.update(style)
    return html.Div([
        html.H3(title, style={
            "margin": "0 0 14px 0", "fontSize": "13px",
            "letterSpacing": "2px", "textTransform": "uppercase",
            "color": THEME["accent_crimson"], "fontWeight": "600",
        }),
        html.Div(children),
    ], style=base)


def _metric(label, value, accent=False):
    return html.Div([
        html.Div(label, style={
            "fontSize": "11px", "letterSpacing": "1.5px",
            "textTransform": "uppercase",
            "color": THEME["secondary_text"],
        }),
        html.Div(value, style={
            "fontSize": "22px", "fontWeight": "600",
            "color": THEME["accent_crimson"] if accent
                    else THEME["primary_text"],
            "marginTop": "4px",
        }),
    ], style={"flex": "1", "minWidth": "110px"})


def _slider(id_, label, min_, max_, value, step, marks=None):
    return html.Div([
        html.Label(label, style={
            "fontSize": "12px", "color": THEME["secondary_text"],
            "letterSpacing": "1px", "textTransform": "uppercase",
        }),
        dcc.Slider(
            id=id_, min=min_, max=max_, value=value, step=step,
            marks=marks or {min_: str(min_), max_: str(max_)},
            tooltip={"always_visible": False},
        ),
    ], style={"marginBottom": "18px"})


def _tab(label, value, content):
    return dcc.Tab(
        label=label, value=value,
        style=dict(
            backgroundColor=THEME["surface"],
            color=THEME["primary_text"], border="none",
            padding="14px 22px", fontSize="11px", letterSpacing="2px",
        ),
        selected_style=dict(
            backgroundColor=THEME["background"],
            color=THEME["accent_crimson"], border="none",
            borderBottom=f"2px solid {THEME['accent_crimson']}",
            padding="14px 22px", fontSize="11px", letterSpacing="2px",
        ),
        children=content,
    )


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(housing: pd.DataFrame, wholesale: pd.DataFrame,
               housing_results: dict, cluster_results: dict,
               explainer):
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Predictive Analytics Lab"
    session_id = str(uuid.uuid4())[:8]

    # -----------------------------------------------------------------
    # Flask routes for the WebXR viewer and its data
    # -----------------------------------------------------------------
    # PROJECT_ROOT resolves to the directory containing main.py,
    # which is where immersive.html and data/ live.
    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    @app.server.route("/immersive")
    def serve_immersive():
        return send_from_directory(PROJECT_ROOT, "immersive.html")

    @app.server.route("/data/<path:filename>")
    def serve_data(filename):
        return send_from_directory(
            os.path.join(PROJECT_ROOT, "data"), filename
        )

    # -----------------------------------------------------------------
    # Precompute visuals
    # -----------------------------------------------------------------
    housing_scatter = housing_scatter_3d(housing)
    cluster_scatter = wholesale_clusters_3d(
        wholesale, cluster_results["labels"],
    )
    surface_fig = housing_surface_3d()
    X_all = housing[HOUSING_FEATURES].values
    y_all = housing["MEDV"].values

    best_name = housing_results["best_model"]
    best = housing_results[best_name]

    # -----------------------------------------------------------------
    # PANEL: DATASET
    # -----------------------------------------------------------------
    dataset_panel = html.Div([
        _card("Housing Dataset", [
            html.Div([
                _metric("Rows", f"{len(housing):,}"),
                _metric("Columns", str(housing.shape[1])),
                _metric("Missing",
                        str(int(housing.isnull().sum().sum()))),
                _metric("Target", "MEDV"),
            ], style={"display": "flex", "gap": "24px",
                       "flexWrap": "wrap"}),
            html.Br(),
            dash_table.DataTable(
                data=housing.head(8).round(3).to_dict("records"),
                columns=[{"name": c, "id": c} for c in housing.columns],
                style_header={
                    "backgroundColor": THEME["surface_alt"],
                    "color": THEME["primary_text"],
                    "fontWeight": "600",
                    "border": f"1px solid {THEME['border']}",
                    "fontSize": "11px", "letterSpacing": "1px",
                },
                style_cell={
                    "backgroundColor": THEME["background"],
                    "color": THEME["primary_text"],
                    "border": f"1px solid {THEME['border']}",
                    "fontSize": "12px", "fontFamily": THEME["font"],
                    "textAlign": "right", "padding": "6px 10px",
                },
                page_size=8,
            ),
        ]),
        _card("Wholesale Dataset", [
            html.Div([
                _metric("Rows", f"{len(wholesale):,}"),
                _metric("Columns", str(wholesale.shape[1])),
                _metric("Missing",
                        str(int(wholesale.isnull().sum().sum()))),
                _metric("Task", "Clustering"),
            ], style={"display": "flex", "gap": "24px",
                       "flexWrap": "wrap"}),
            html.Br(),
            dash_table.DataTable(
                data=wholesale.head(8).to_dict("records"),
                columns=[{"name": c, "id": c} for c in wholesale.columns],
                style_header={
                    "backgroundColor": THEME["surface_alt"],
                    "color": THEME["primary_text"],
                    "fontWeight": "600",
                    "border": f"1px solid {THEME['border']}",
                    "fontSize": "11px", "letterSpacing": "1px",
                },
                style_cell={
                    "backgroundColor": THEME["background"],
                    "color": THEME["primary_text"],
                    "border": f"1px solid {THEME['border']}",
                    "fontSize": "12px", "fontFamily": THEME["font"],
                    "textAlign": "right", "padding": "6px 10px",
                },
                page_size=8,
            ),
        ]),
    ])

    # -----------------------------------------------------------------
    # PANEL: VISUALIZATION
    # -----------------------------------------------------------------
    visualization_panel = html.Div([
        _card("3D Housing Scatter — rotate, zoom, hover",
              dcc.Graph(figure=housing_scatter,
                        config={"displayModeBar": True,
                                "displaylogo": False})),
        _card("Prediction Surface — best model over MedInc × AveOccup",
              dcc.Graph(figure=surface_fig,
                        config={"displayModeBar": True,
                                "displaylogo": False})),
    ])

    # -----------------------------------------------------------------
    # PANEL: MACHINE LEARNING
    # -----------------------------------------------------------------
    model_names = [k for k in housing_results if k != "best_model"]
    comparison_rows = []
    for name in model_names:
        r = housing_results[name]
        comparison_rows.append({
            "Model": name,
            "R² mean": f"{r['r2_mean']:.3f}",
            "R² std": f"{r['r2_std']:.3f}",
            "RMSE": f"{r['rmse_mean']:.3f}",
            "MAE": f"{r['mae_mean']:.3f}",
        })

    elbow_fig = go.Figure(go.Scatter(
        x=cluster_results["ks"], y=cluster_results["inertias"],
        mode="lines+markers",
        line=dict(color=THEME["accent_crimson"], width=2),
        marker=dict(size=8),
    ))
    elbow_fig.update_layout(**PLOTLY_LAYOUT)
    elbow_fig.update_layout(
        title="K-Means Elbow Curve (Wholesale)",
        xaxis_title="k", yaxis_title="Inertia", height=350,
    )

    ml_panel = html.Div([
        _card("Model Comparison — 5-fold Cross-Validation", [
            dash_table.DataTable(
                data=comparison_rows,
                columns=[{"name": c, "id": c} for c in
                         ["Model", "R² mean", "R² std", "RMSE", "MAE"]],
                style_header={
                    "backgroundColor": THEME["surface_alt"],
                    "color": THEME["primary_text"],
                    "fontWeight": "600", "fontSize": "11px",
                },
                style_cell={
                    "backgroundColor": THEME["background"],
                    "color": THEME["primary_text"],
                    "fontSize": "12px", "padding": "8px 12px",
                },
                style_data_conditional=[{
                    "if": {"filter_query":
                           f'{{Model}} = "{best_name}"'},
                    "backgroundColor": "#1a0a0a",
                    "color": THEME["accent_crimson"],
                }],
            ),
        ]),
        _card("K-Means — Wholesale", [
            html.Div([
                _metric("Silhouette",
                        f"{cluster_results['silhouette']:.3f}",
                        accent=True),
                _metric("k", str(cluster_results["k"])),
                _metric("Samples", f"{len(wholesale):,}"),
            ], style={"display": "flex", "gap": "24px"}),
            html.Br(),
            dcc.Graph(figure=elbow_fig,
                      config={"displayModeBar": False}),
        ]),
    ])

    # -----------------------------------------------------------------
    # PANEL: EXPLAINABILITY
    # -----------------------------------------------------------------
    explain_panel = html.Div([
        _card("SHAP — Global and Local", [
            dcc.RadioItems(
                id="xai-view",
                options=[
                    {"label": " Global (beeswarm)", "value": "global"},
                    {"label": " Local (waterfall for current sliders)",
                     "value": "local"},
                ],
                value="global",
                style={"color": THEME["primary_text"], "fontSize": "12px",
                       "marginBottom": "14px"},
            ),
            html.Div(id="xai-plot"),
        ]),
        _card("Counterfactual Explorer", [
            html.P(
                "Set a target prediction. The system finds the smallest "
                "input change that would reach it.",
                style={"color": THEME["secondary_text"],
                       "fontSize": "12px"},
            ),
            dcc.Slider(id="cf-target", min=0.5, max=5.0, value=3.0,
                       step=0.1,
                       marks={i: str(i) for i in range(1, 6)}),
            html.Button(
                "Compute Counterfactual", id="cf-compute", n_clicks=0,
                style={
                    "backgroundColor": THEME["accent_crimson"],
                    "color": "#fff", "border": "none",
                    "padding": "10px 22px", "fontSize": "11px",
                    "letterSpacing": "2px",
                    "textTransform": "uppercase",
                    "cursor": "pointer", "marginTop": "14px",
                    "borderRadius": "3px",
                },
            ),
            html.Div(id="cf-result", style={"marginTop": "18px"}),
        ]),
    ])

    # -----------------------------------------------------------------
    # PANEL: UNCERTAINTY
    # -----------------------------------------------------------------
    uncertainty_panel = html.Div([
        _card("Conformal Prediction Intervals", [
            html.P(
                "Split conformal prediction provides finite-sample "
                "coverage guarantees. The 90% interval shown below is "
                "calibrated on held-out data.",
                style={"color": THEME["secondary_text"],
                       "fontSize": "12px"},
            ),
            html.Div(id="conformal-output"),
        ]),
    ])

    # -----------------------------------------------------------------
    # PANEL: PREDICTION
    # -----------------------------------------------------------------
    prediction_panel = html.Div([
        _card("Interactive Prediction Console", [
            html.P(
                "Adjust the feature sliders. The prediction updates in "
                "real time. This is the core educational interaction.",
                style={"color": THEME["secondary_text"],
                       "fontSize": "12px", "marginBottom": "20px"},
            ),
            html.Div([
                html.Div([
                    _slider("slider-medinc", "Median Income",
                            0.5, 15.0, 5.0, 0.1),
                    _slider("slider-houseage", "House Age",
                            1, 52, 28, 1),
                ], style={"flex": "1"}),
                html.Div([
                    _slider("slider-averooms", "Average Rooms",
                            2.0, 10.0, 5.4, 0.1),
                    _slider("slider-aveoccup", "Average Occupancy",
                            1.0, 8.0, 3.0, 0.1),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "32px"}),
            html.Div(id="prediction-output", style={
                "marginTop": "18px", "padding": "22px",
                "backgroundColor": THEME["background"],
                "border": f"1px solid {THEME['accent_crimson']}",
                "borderRadius": "4px", "textAlign": "center",
            }),
        ]),
    ])

    # -----------------------------------------------------------------
    # PANEL: IMMERSIVE
    # -----------------------------------------------------------------
    immersive_panel = html.Div([
        _card("Immersive Mode (WebXR)", [
            html.P(
                "The immersive viewer renders the dataset as a 3D point "
                "cloud in the browser. On a WebXR-capable headset "
                "(Quest, Vision Pro), press the 'Enter VR' button "
                "inside the viewer. On desktop, use mouse and scroll "
                "to orbit.",
                style={"color": THEME["secondary_text"],
                       "fontSize": "12px"},
            ),
            html.A(
                "Open Immersive Viewer",
                href="/immersive",
                target="_blank",
                style={
                    "display": "inline-block",
                    "backgroundColor": THEME["accent_crimson"],
                    "color": "#fff", "padding": "12px 26px",
                    "textDecoration": "none", "fontSize": "11px",
                    "letterSpacing": "2px",
                    "textTransform": "uppercase",
                    "borderRadius": "3px",
                },
            ),
        ]),
    ])

    # -----------------------------------------------------------------
    # PANEL: CASE STUDIES
    # -----------------------------------------------------------------
    case_panel = html.Div([
        _card("Case Study 1 — Wholesale Customer Segmentation", [
            html.P(
                "A wholesale distributor segments customers by "
                "purchasing behaviour. Students run K-Means (k=4) and "
                "explore the resulting groups.",
                style={"color": THEME["secondary_text"],
                       "fontSize": "13px"},
            ),
            dcc.Graph(figure=cluster_scatter,
                      config={"displayModeBar": True,
                              "displaylogo": False}),
        ]),
        _card("Case Study 2 — Housing Price Analysis", [
            html.P(
                "Estimate median home value from socio-economic "
                "features. Compare models, inspect SHAP, and use the "
                "counterfactual explorer to find decision boundaries.",
                style={"color": THEME["secondary_text"],
                       "fontSize": "13px"},
            ),
        ]),
    ])

    # -----------------------------------------------------------------
    # PANEL: ASSESSMENT (adaptive)
    # -----------------------------------------------------------------
    assessment_panel = html.Div([
        _card("Adaptive Assessment (Bayesian Knowledge Tracing)", [
            html.P(
                "Click START to begin. The engine selects each next "
                "question based on your current concept mastery. "
                "Answer to update the model and see your mastery "
                "report.",
                style={"color": THEME["secondary_text"],
                       "fontSize": "12px"},
            ),
            html.Button(
                "START ASSESSMENT", id="start-assessment", n_clicks=0,
                style={
                    "backgroundColor": THEME["accent_crimson"],
                    "color": "#fff", "border": "none",
                    "padding": "12px 28px", "fontSize": "12px",
                    "letterSpacing": "2px",
                    "textTransform": "uppercase",
                    "cursor": "pointer", "borderRadius": "3px",
                },
            ),
            html.Div(id="assessment-q", style={"marginTop": "20px"}),
            html.Div(id="assessment-report",
                     style={"marginTop": "20px"}),
        ]),
    ])

    # -----------------------------------------------------------------
    # PANEL: ANALYTICS
    # -----------------------------------------------------------------
    analytics_panel = build_analytics_panel(session_id)

    # -----------------------------------------------------------------
    # Assembled layout
    # -----------------------------------------------------------------
    app.layout = html.Div([
        html.Div([
            html.H1("PREDICTIVE ANALYTICS LAB",
                    style={
                        "margin": "0", "fontSize": "20px",
                        "letterSpacing": "4px", "fontWeight": "600",
                        "color": THEME["primary_text"],
                    }),
            html.Div(
                f"VR-POWERED · IMMERSIVE ANALYTICS · "
                f"BEST MODEL: {best_name.upper()} "
                f"(R²={best['r2_mean']:.3f})",
                style={
                    "fontSize": "10px", "letterSpacing": "3px",
                    "color": THEME["accent_crimson"],
                    "marginTop": "4px",
                },
            ),
        ], style={
            "padding": "22px 32px",
            "borderBottom": f"1px solid {THEME['border']}",
            "backgroundColor": THEME["surface"],
        }),

        dcc.Tabs(
            id="main-tabs", value="tab-dataset",
            colors={
                "border": THEME["border"],
                "primary": THEME["accent_crimson"],
                "background": THEME["surface"],
            },
            children=[
                _tab("DATASET", "tab-dataset", dataset_panel),
                _tab("VISUALIZATION", "tab-viz", visualization_panel),
                _tab("ML", "tab-ml", ml_panel),
                _tab("EXPLAINABILITY", "tab-xai", explain_panel),
                _tab("UNCERTAINTY", "tab-unc", uncertainty_panel),
                _tab("PREDICTION", "tab-pred", prediction_panel),
                _tab("IMMERSIVE", "tab-imm", immersive_panel),
                _tab("CASE STUDIES", "tab-cases", case_panel),
                _tab("ASSESSMENT", "tab-assess", assessment_panel),
                _tab("ANALYTICS", "tab-analytics", analytics_panel),
            ],
            style={"backgroundColor": THEME["surface"]},
        ),
        html.Div(style={"height": "32px"}),
    ], style={
        "backgroundColor": THEME["background"],
        "minHeight": "100vh",
        "fontFamily": THEME["font"],
        "padding": "0 0 32px 0",
    })

    # -----------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------

    @app.callback(
        Output("prediction-output", "children"),
        [Input("slider-medinc", "value"),
         Input("slider-houseage", "value"),
         Input("slider-averooms", "value"),
         Input("slider-aveoccup", "value")],
    )
    def update_prediction(medinc, houseage, averooms, aveoccup):
        pred = predict_housing(medinc, houseage, averooms, aveoccup)
        return html.Div([
            html.Div("PREDICTED MEDIAN VALUE",
                     style={"fontSize": "10px", "letterSpacing": "3px",
                            "color": THEME["secondary_text"]}),
            html.Div(f"${pred * 100_000:,.0f}",
                     style={"fontSize": "34px", "fontWeight": "600",
                            "color": THEME["accent_crimson"],
                            "marginTop": "6px"}),
        ])

    @app.callback(
        Output("xai-plot", "children"),
        [Input("xai-view", "value"),
         Input("slider-medinc", "value"),
         Input("slider-houseage", "value"),
         Input("slider-averooms", "value"),
         Input("slider-aveoccup", "value")],
    )
    def update_xai(view, medinc, houseage, averooms, aveoccup):
        try:
            if view == "global":
                b64 = explainer.global_summary(X_all)
            else:
                x = [medinc, houseage, averooms, aveoccup]
                b64 = explainer.local_waterfall(x)
            return html.Img(
                src=f"data:image/png;base64,{b64}",
                style={"width": "100%", "borderRadius": "3px"},
            )
        except Exception as e:
            return html.Div(
                f"SHAP rendering failed: {e}",
                style={"color": THEME["accent_crimson"],
                       "fontSize": "12px"},
            )

    @app.callback(
        Output("cf-result", "children"),
        Input("cf-compute", "n_clicks"),
        [State("cf-target", "value"),
         State("slider-medinc", "value"),
         State("slider-houseage", "value"),
         State("slider-averooms", "value"),
         State("slider-aveoccup", "value")],
    )
    def compute_cf(n_clicks, target, medinc, houseage, averooms,
                   aveoccup):
        if not n_clicks:
            return ""
        m, _ = _best_model()
        if m is None:
            return html.Div("No model available.")
        current = [medinc, houseage, averooms, aveoccup]
        try:
            cf = counterfactual_regression(
                m, current, HOUSING_FEATURES, FEATURE_BOUNDS, target,
            )
        except Exception as e:
            return html.Div(f"Counterfactual failed: {e}")
        deltas = cf["deltas"]
        rows = [html.Div([
            html.Span(f"{k}: ", style={"fontWeight": "600"}),
            html.Span(f"{v:+.2f}",
                      style={"color": THEME["accent_crimson"]}),
        ], style={"fontSize": "13px", "marginBottom": "4px"})
            for k, v in deltas.items()]
        return html.Div([
            html.Div(
                f"Current prediction: "
                f"${cf['current_prediction'] * 100_000:,.0f}",
                style={"fontSize": "12px"},
            ),
            html.Div(
                f"Counterfactual prediction: "
                f"${cf['counterfactual_prediction'] * 100_000:,.0f}",
                style={"fontSize": "12px",
                       "color": THEME["accent_crimson"]},
            ),
            html.Div(f"Distance: {cf['distance']:.3f}",
                     style={"fontSize": "11px",
                            "color": THEME["secondary_text"]}),
            html.Hr(style={"borderColor": THEME["border"]}),
            html.Div("Feature deltas:",
                     style={"fontSize": "11px",
                            "letterSpacing": "1.5px",
                            "textTransform": "uppercase",
                            "color": THEME["secondary_text"]}),
            *rows,
        ])

    @app.callback(
        Output("conformal-output", "children"),
        [Input("slider-medinc", "value"),
         Input("slider-houseage", "value"),
         Input("slider-averooms", "value"),
         Input("slider-aveoccup", "value")],
    )
    def update_conformal(medinc, houseage, averooms, aveoccup):
        m, _ = _best_model()
        if m is None:
            return html.Div("No model available.")
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.4, random_state=42,
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_te, y_te, test_size=0.5, random_state=42,
        )
        x = np.array([[medinc, houseage, averooms, aveoccup]])
        lo, hi, pred, q = conformal_intervals(
            m, X_cal, y_cal, x, alpha=0.10,
        )
        return html.Div([
            html.Div([
                _metric("Point prediction",
                        f"${pred[0] * 100_000:,.0f}", accent=True),
                _metric("90% interval",
                        f"${lo[0] * 100_000:,.0f} – "
                        f"${hi[0] * 100_000:,.0f}"),
                _metric("Interval half-width",
                        f"${q * 100_000:,.0f}"),
            ], style={"display": "flex", "gap": "24px",
                       "flexWrap": "wrap"}),
        ])

    # -----------------------------------------------------------------
    # Adaptive assessment state (per app instance)
    # -----------------------------------------------------------------
    state = {"engine": None, "current_q": None}

    def _render_next_question():
        engine = state["engine"]
        if engine is None or engine.is_complete(8):
            report = engine.mastery_report() if engine else {}
            return html.Div([
                html.Div("Assessment complete. Mastery report:",
                         style={"fontSize": "13px",
                                "marginBottom": "10px"}),
                *[html.Div(f"{k}: {v:.3f}",
                           style={"fontSize": "12px",
                                  "color": THEME["primary_text"]})
                  for k, v in report.items()],
            ])
        q = engine.next_question()
        state["current_q"] = q
        return html.Div([
            html.Div(f"Concept: {q['concept']} · "
                     f"Difficulty: {q['difficulty']}",
                     style={"fontSize": "10px",
                            "letterSpacing": "1.5px",
                            "color": THEME["secondary_text"]}),
            html.Div(q["text"], style={"fontSize": "14px",
                                        "margin": "8px 0 12px 0"}),
            dcc.RadioItems(
                id="q-options",
                options=[{"label": f"  {o}", "value": o}
                         for o in q["options"]],
                style={"fontSize": "13px",
                       "color": THEME["primary_text"]},
                labelStyle={"display": "block", "padding": "4px 0"},
                inputStyle={"marginRight": "8px"},
            ),
            html.Button("SUBMIT", id="submit-answer", n_clicks=0,
                        style={
                            "backgroundColor": THEME["accent_crimson"],
                            "color": "#fff", "border": "none",
                            "padding": "10px 22px", "fontSize": "11px",
                            "letterSpacing": "2px",
                            "textTransform": "uppercase",
                            "cursor": "pointer", "marginTop": "12px",
                            "borderRadius": "3px",
                        }),
        ])

    @app.callback(
        Output("assessment-q", "children"),
        Input("start-assessment", "n_clicks"),
        prevent_initial_call=True,
    )
    def start_assessment(n):
        if not n:
            return ""
        state["engine"] = AdaptiveAssessment(QUESTION_BANK)
        log_event(session_id, "assessment_start")
        return _render_next_question()

    @app.callback(
        Output("assessment-report", "children"),
        Output("assessment-q", "children", allow_duplicate=True),
        Input("submit-answer", "n_clicks"),
        State("q-options", "value"),
        prevent_initial_call=True,
    )
    def submit_answer(n, value):
        if not n or state["engine"] is None or state["current_q"] is None:
            return "", dash.no_update
        q = state["current_q"]
        correct = (value == q["answer"])
        state["engine"].submit(q["id"], correct)
        log_event(session_id, "assessment_answer",
                  {"qid": q["id"], "correct": correct})

        feedback = html.Div([
            html.Div(("✓ Correct" if correct else "✗ Incorrect"),
                     style={"color": THEME["success"] if correct
                            else THEME["accent_crimson"],
                            "fontWeight": "600", "fontSize": "13px"}),
            html.Div(q["feedback"],
                     style={"fontSize": "12px",
                            "color": THEME["secondary_text"],
                            "marginTop": "4px"}),
        ])

        next_q = _render_next_question()
        return feedback, next_q

    return app