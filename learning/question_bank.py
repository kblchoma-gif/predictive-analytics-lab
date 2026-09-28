"""Adaptive question bank tagged by concept and difficulty."""

QUESTION_BANK = [
    {
        "id": "q1", "concept": "feature_importance", "difficulty": "easy",
        "text": "Which variable shows the strongest relationship with MEDV?",
        "options": ["MedInc", "HouseAge", "AveOccup", "Population"],
        "answer": "MedInc",
        "feedback": "Median income dominates. SHAP confirms this globally.",
    },
    {
        "id": "q2", "concept": "feature_importance", "difficulty": "medium",
        "text": ("In the SHAP beeswarm plot, what does a point's "
                 "horizontal position represent?"),
        "options": [
            "The feature's contribution to that prediction",
            "The feature's raw value",
            "The model's confidence",
            "The training-set average",
        ],
        "answer": "The feature's contribution to that prediction",
        "feedback": "SHAP values quantify each feature's push on the prediction.",
    },
    {
        "id": "q3", "concept": "uncertainty", "difficulty": "easy",
        "text": "What does a 90% conformal prediction interval mean?",
        "options": [
            "90% of future true values will fall inside the interval",
            "The model is 90% accurate",
            "90% of training data was used",
            "The prediction is wrong 10% of the time",
        ],
        "answer": "90% of future true values will fall inside the interval",
        "feedback": "Conformal intervals have finite-sample coverage guarantees.",
    },
    {
        "id": "q4", "concept": "uncertainty", "difficulty": "hard",
        "text": ("Two models have identical R². One has wider conformal "
                 "intervals. What does this indicate?"),
        "options": [
            "Different residual distributions — the wider one is less "
            "reliable per-point",
            "The wider one is always worse",
            "R² is the only metric that matters",
            "They are identical in every way",
        ],
        "answer": ("Different residual distributions — the wider one is "
                   "less reliable per-point"),
        "feedback": "R² is an aggregate; conformal widths are local.",
    },
    {
        "id": "q5", "concept": "counterfactual", "difficulty": "medium",
        "text": "A counterfactual tells you:",
        "options": [
            "The nearest input change that would flip the prediction",
            "The exact true value",
            "The model's architecture",
            "Nothing useful",
        ],
        "answer": "The nearest input change that would flip the prediction",
        "feedback": "Counterfactuals reveal decision boundaries concretely.",
    },
    {
        "id": "q6", "concept": "counterfactual", "difficulty": "hard",
        "text": "Why might a counterfactual suggest an unrealistic change?",
        "options": [
            "The optimizer only sees the model, not real-world feasibility",
            "Because the model is always wrong",
            "Because counterfactuals are random",
            "Because R² is low",
        ],
        "answer": "The optimizer only sees the model, not real-world feasibility",
        "feedback": "This is a key limitation of post-hoc explanation methods.",
    },
    {
        "id": "q7", "concept": "model_comparison", "difficulty": "easy",
        "text": ("Which is typically true of Random Forest vs Linear "
                 "Regression on housing data?"),
        "options": [
            "RF captures non-linearities and usually improves R²",
            "Linear always wins",
            "They are always identical",
            "RF is always worse",
        ],
        "answer": "RF captures non-linearities and usually improves R²",
        "feedback": "Tree ensembles model non-linearity automatically.",
    },
    {
        "id": "q8", "concept": "model_comparison", "difficulty": "medium",
        "text": "Cross-validation is preferred over a single train/test split because:",
        "options": [
            "It gives a more reliable estimate of generalisation",
            "It is faster",
            "It uses less data",
            "It avoids overfitting entirely",
        ],
        "answer": "It gives a more reliable estimate of generalisation",
        "feedback": "CV averages over multiple splits, reducing variance.",
    },
    {
        "id": "q9", "concept": "ethics", "difficulty": "easy",
        "text": "Why should predictions not be treated as facts?",
        "options": [
            "They are statistical estimates with uncertainty",
            "Because models are always wrong",
            "Because data is fake",
            "They should be treated as facts",
        ],
        "answer": "They are statistical estimates with uncertainty",
        "feedback": "Predictions inform decisions; they do not determine them.",
    },
    {
        "id": "q10", "concept": "ethics", "difficulty": "hard",
        "text": "A model trained on historical data may perpetuate bias because:",
        "options": [
            "It learns whatever patterns exist, including discriminatory ones",
            "It is always fair",
            "Bias only exists in deep learning",
            "Data is never biased",
        ],
        "answer": "It learns whatever patterns exist, including discriminatory ones",
        "feedback": "This is why ethical review of datasets is essential.",
    },
]