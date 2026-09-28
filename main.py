"""
Advanced entry point for the VR-Powered Predictive Analytics Lab.

Responsibilities
----------------
1. Load the two datasets (California Housing, Wholesale Customers).
2. Train the six-model zoo with 5-fold cross-validation + tuning.
3. Select a "best model" with a bias toward SHAP-friendly families
   when the accuracy gap is small (keeps the explainability panel fast).
4. Fit a SHAP explainer on the selected model.
5. Train the wholesale K-Means clustering baseline.
6. Export the immersive point cloud for the WebXR viewer.
7. Launch the Dash dashboard.
"""
import os
import numpy as np

from data.loader import load_housing, load_wholesale
from models.advanced_train import evaluate_all_models
from models.explainability import ModelExplainer
from models.train import train_wholesale_clusters
from visualization.export_immersive import export_housing_points
from ui.dashboard import create_app


# Features used for housing regression throughout the project.
HOUSING_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveOccup"]
HOUSING_TARGET = "MEDV"

# Model families that TreeExplainer / LinearExplainer can handle directly.
# Used by the best-model selection heuristic below.
SHAP_FRIENDLY = {
    "XGBoost", "GradientBoosting", "RandomForest", "Ridge", "Linear",
}

# If a non-tree model beats the best tree model by more than this margin
# (in mean cross-validated R²), we accept the slower SHAP path.
# Otherwise we keep the tree model for responsiveness.
R2_TOLERANCE = 0.02


def _select_best_model(results: dict) -> str:
    """
    Choose the model to explain and to use for live predictions.

    Preference order:
      1. If the overall best model is SHAP-friendly, use it.
      2. Otherwise, if the best SHAP-friendly model is within
         R2_TOLERANCE of the overall best, use the SHAP-friendly one.
      3. Otherwise, use the overall best (accept the slower SHAP path).
    """
    model_names = [k for k in results if k != "best_model"]
    overall_best = max(model_names, key=lambda k: results[k]["r2_mean"])

    if overall_best in SHAP_FRIENDLY:
        return overall_best

    friendly_candidates = [k for k in model_names if k in SHAP_FRIENDLY]
    if not friendly_candidates:
        return overall_best

    best_friendly = max(
        friendly_candidates, key=lambda k: results[k]["r2_mean"],
    )
    gap = results[overall_best]["r2_mean"] - results[best_friendly]["r2_mean"]

    if gap <= R2_TOLERANCE:
        print(
            f"[main] Preferring {best_friendly} over {overall_best} "
            f"(R² gap {gap:+.3f} ≤ {R2_TOLERANCE}); "
            f"tree-friendly SHAP path."
        )
        return best_friendly

    print(
        f"[main] Keeping {overall_best} despite slower SHAP path "
        f"(R² gap {gap:+.3f} > {R2_TOLERANCE})."
    )
    return overall_best


def main():
    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    print("[main] Loading datasets...")
    housing = load_housing()
    wholesale = load_wholesale()
    print(f"[main] Housing: {housing.shape} | "
          f"Wholesale: {wholesale.shape}")

    # ------------------------------------------------------------------
    # 2. Train the model zoo with cross-validation
    # ------------------------------------------------------------------
    print("[main] Training model zoo with 5-fold CV and tuning...")
    results = evaluate_all_models(housing, n_splits=5, tune=True)

    # ------------------------------------------------------------------
    # 3. Select the best model (with SHAP-friendliness bias)
    # ------------------------------------------------------------------
    best_name = _select_best_model(results)
    results["best_model"] = best_name
    best_stats = results[best_name]
    print(f"[main] Best model: {best_name}  "
          f"(R² = {best_stats['r2_mean']:.3f} ± "
          f"{best_stats['r2_std']:.3f})")

    # ------------------------------------------------------------------
    # 4. Build the SHAP explainer
    # ------------------------------------------------------------------
    print("[main] Fitting SHAP explainer on best model...")
    X = housing[HOUSING_FEATURES].values
    best_pipeline = results[best_name]["model"]
    underlying = best_pipeline.named_steps["model"]

    explainer = ModelExplainer(
        underlying,
        HOUSING_FEATURES,
        background_data=X,
    )

    # ------------------------------------------------------------------
    # 5. Train the wholesale K-Means baseline
    # ------------------------------------------------------------------
    print("[main] Training wholesale K-Means clustering...")
    cluster_results = train_wholesale_clusters(wholesale, k=4)
    print(f"[main] Wholesale silhouette: "
          f"{cluster_results['silhouette']:.3f}")

    # ------------------------------------------------------------------
    # 6. Export the immersive point cloud
    # ------------------------------------------------------------------
    print("[main] Exporting immersive point cloud...")
    export_path = export_housing_points(housing)
    print(f"[main] Wrote {export_path}")

    # ------------------------------------------------------------------
    # 7. Launch the dashboard
    # ------------------------------------------------------------------
    print("[main] Launching dashboard on http://127.0.0.1:8050")
    app = create_app(
        housing=housing,
        wholesale=wholesale,
        housing_results=results,
        cluster_results=cluster_results,
        explainer=explainer,
    )
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()