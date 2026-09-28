"""
Pre/post knowledge test.

Reference: Hake (1998) for normalised gain.
"""
PRE_TEST = [
    {"q": "What is R²?", "options": [
        "Proportion of variance explained",
        "Model accuracy",
        "Number of features",
        "Training time",
    ], "answer": "Proportion of variance explained"},
    {"q": "What does a prediction interval represent?", "options": [
        "Range likely to contain the true value",
        "Model error",
        "Feature count",
        "None of these",
    ], "answer": "Range likely to contain the true value"},
    {"q": "What is a counterfactual in ML?", "options": [
        "Smallest input change that flips a prediction",
        "The true value",
        "A model type",
        "A training algorithm",
    ], "answer": "Smallest input change that flips a prediction"},
]

POST_TEST = [
    {"q": "Which best describes SHAP values?", "options": [
        "Per-feature contributions to a prediction",
        "Model accuracy scores",
        "Training-set sizes",
        "Feature counts",
    ], "answer": "Per-feature contributions to a prediction"},
    {"q": "Conformal prediction intervals guarantee:", "options": [
        "Finite-sample coverage of future values",
        "Zero error",
        "Perfect calibration",
        "None of these",
    ], "answer": "Finite-sample coverage of future values"},
    {"q": "Why is cross-validation preferred over a single split?",
     "options": [
        "More reliable generalisation estimate",
        "Faster",
        "Simpler",
        "No reason",
    ], "answer": "More reliable generalisation estimate"},
]


def compute_normalised_gain(pre_score, post_score, max_score) -> float:
    """Hake's normalised gain: <g> = (post - pre) / (max - pre)."""
    if max_score == pre_score:
        return 0.0
    return (post_score - pre_score) / (max_score - pre_score)