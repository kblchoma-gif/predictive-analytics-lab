"""
Export a sampled, normalised point cloud for the WebXR viewer.
"""
import json
import os
import numpy as np


OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "immersive_points.json",
)


def export_housing_points(df, x="MedInc", y="AveRooms", z="MEDV",
                          target="MEDV", max_points: int = 1200) -> str:
    d = df.sample(min(len(df), max_points), random_state=42).copy()

    def norm(s):
        lo, hi = s.min(), s.max()
        if hi == lo:
            return np.zeros(len(s))
        return (s - lo) / (hi - lo)

    xs = norm(d[x]) * 10 - 5
    ys = norm(d[y]) * 6
    zs = norm(d[z]) * 10 - 5
    vals = norm(d[target])

    points = [
        {"x": float(a), "y": float(b), "z": float(c), "value": float(v)}
        for a, b, c, v in zip(xs, ys, zs, vals)
    ]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(points, f)
    return OUT_PATH