"""
Dataset loading with deterministic synthetic fallbacks.

Primary datasets:
  - California Housing (sklearn built-in) — replaces Boston Housing,
    which was removed from sklearn for ethical reasons.
  - Wholesale Customers (UCI ML Repository, direct CSV download).

If the network is unavailable, deterministic synthetic data with the
same schema is generated (seed=42).
"""
import io
import urllib.request
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


WHOLESALE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00292/Wholesale%20customers%20data.csv"
)


# ---------------------------------------------------------------------------
# Wholesale Customers
# ---------------------------------------------------------------------------

def _synthetic_wholesale(n: int = 440, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Channel": rng.choice([1, 2], size=n, p=[0.65, 0.35]),
        "Region": rng.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2]),
        "Fresh": rng.gamma(2.0, 6000, n).astype(int),
        "Milk": rng.gamma(2.0, 3000, n).astype(int),
        "Grocery": rng.gamma(2.0, 4000, n).astype(int),
        "Frozen": rng.gamma(2.0, 1500, n).astype(int),
        "Detergents_Paper": rng.gamma(2.0, 1500, n).astype(int),
        "Delicassen": rng.gamma(2.0, 1000, n).astype(int),
    })


def load_wholesale(force_synthetic: bool = False) -> pd.DataFrame:
    """Load Wholesale Customers from UCI; fall back to synthetic."""
    if not force_synthetic:
        try:
            with urllib.request.urlopen(WHOLESALE_URL, timeout=10) as r:
                raw = r.read()
            df = pd.read_csv(io.BytesIO(raw))
            df.columns = [c.strip() for c in df.columns]
            print("[loader] Wholesale Customers loaded from UCI.")
            return df
        except Exception as e:
            print(f"[loader] Wholesale download failed ({e}); "
                  f"using synthetic.")
    return _synthetic_wholesale()


# ---------------------------------------------------------------------------
# Housing
# ---------------------------------------------------------------------------

def _synthetic_housing(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Synthetic California-like housing with correct schema."""
    rng = np.random.default_rng(seed)
    medinc = rng.gamma(3, 1.2, n)
    houseage = rng.uniform(1, 52, n)
    averooms = rng.normal(5.4, 1.2, n).clip(2, 10)
    avebedrms = rng.normal(1.1, 0.2, n).clip(0.5, 2)
    population = rng.gamma(3, 400, n)
    aveoccup = rng.gamma(3, 1.0, n).clip(1, 8)
    latitude = rng.uniform(32, 42, n)
    longitude = rng.uniform(-124, -114, n)
    medv = (
        0.5 * medinc + 0.2 * averooms - 0.1 * aveoccup
        + rng.normal(0, 0.3, n)
    ).clip(0.15, 5.0)
    return pd.DataFrame({
        "MedInc": medinc, "HouseAge": houseage,
        "AveRooms": averooms, "AveBedrms": avebedrms,
        "Population": population, "AveOccup": aveoccup,
        "Latitude": latitude, "Longitude": longitude,
        "MEDV": medv,
    })


def load_housing(force_synthetic: bool = False) -> pd.DataFrame:
    """Load California Housing; fall back to synthetic."""
    if not force_synthetic:
        try:
            data = fetch_california_housing(as_frame=True)
            df = data.frame.copy().rename(columns={"MedHouseVal": "MEDV"})
            print("[loader] California Housing loaded from sklearn.")
            return df
        except Exception as e:
            print(f"[loader] Housing fetch failed ({e}); "
                  f"using synthetic.")
    return _synthetic_housing()