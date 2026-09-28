# VR-Powered Predictive Analytics Lab

**A one-day academic prototype with advanced ML, XAI, adaptive assessment,
and browser-based VR.**

## Features

- 6 regression models with 5-fold cross-validation (Linear, Ridge, RF,
  GradientBoosting, XGBoost, MLP).
- SHAP global + local explanations, LIME, counterfactual generator.
- Conformal prediction intervals (90% coverage guarantees).
- Bayesian Knowledge Tracing for adaptive assessment.
- Learning analytics via SQLite event log.
- WebXR immersive mode (runs on Quest / Vision Pro / desktop fallback).

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Open <http://127.0.0.1:8050>. Immersive mode: open `immersive.html`
in a WebXR-capable browser (or use the "Open Immersive" link inside the
dashboard).

## Test

```bash
pytest -v
```

## Layout

- `data/` — dataset loaders (with synthetic fallback).
- `models/` — training, prediction, XAI, uncertainty, counterfactuals.
- `visualization/` — Plotly 3D charts and point-cloud exporter.
- `learning/` — BKT, question bank, pre/post test.
- `analytics/` — SQLite event logger.
- `evaluation/` — SUS instrument.
- `ui/` — Dash dashboard and theme.
- `immersive.html` — WebXR viewer.

## Datasets

- **California Housing** (sklearn built-in). Replaces Boston Housing,
  which was removed from sklearn for ethical reasons.
- **Wholesale Customers** (UCI). Auto-download with synthetic fallback.

## Limitations

Desktop + WebXR only. No native headset app. No external user testing.
Predictions are statistical estimates, not facts.