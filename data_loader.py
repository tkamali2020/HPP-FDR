"""
data_loader.py
===============
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from config import CONFIG


_DEFAULT_DATA_PATH = str(Path(__file__).parent / "data" / "heart_disease.csv")


_RAW_UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]

_SELECTED_RAW_COLUMNS = ["age", "thal", "trestbps", "chol", "cp"]


@dataclass
class HeartDiseaseData:
    X_train: np.ndarray            
    X_test: np.ndarray
    y_train: np.ndarray            
    y_test: np.ndarray
    X_train_norm: np.ndarray       
    X_test_norm: np.ndarray
    feature_names: list
    feature_min: np.ndarray        
    feature_max: np.ndarray


# --------------------------------------------------------------------------- #
def _load_dataframe(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"'{path}' not found. Download the UCI Heart Disease dataset "
            f"(https://archive.ics.uci.edu/dataset/45/heart+disease) and "
            f"place it at this path (raw 'processed.cleveland.data' or a "
            f"headered CSV mirror both work)."
        )

    
    with open(path, "r") as f:
        first_line = f.readline()

    looks_headerless = all(
        tok.strip().replace(".", "").replace("-", "").replace("?", "0").isdigit()
        for tok in first_line.strip().split(",") if tok.strip() != ""
    )

    if looks_headerless:
        df = pd.read_csv(path, header=None, names=_RAW_UCI_COLUMNS, na_values="?")
    else:
        df = pd.read_csv(path, na_values="?")
        df.columns = [c.strip().lower() for c in df.columns]
        if "num" not in df.columns and "target" in df.columns:
            df = df.rename(columns={"target": "num"})
    return df


# --------------------------------------------------------------------------- #
def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    return df


# --------------------------------------------------------------------------- #
def _build_urgency_proxy_target(df: pd.DataFrame) -> np.ndarray:
    
    return (df["num"].clip(0, 4) / 4.0).to_numpy()


# --------------------------------------------------------------------------- #
def load_and_prepare(path: str = None,
                      seed: int = None) -> HeartDiseaseData:
    path = path or _DEFAULT_DATA_PATH
    cfg = CONFIG.fuzzy
    rng = np.random.default_rng(seed or CONFIG.env.random_seed)

    df = _load_dataframe(path)
    df = _impute_missing(df)

    X = df[_SELECTED_RAW_COLUMNS].to_numpy(dtype=np.float64)
    y = _build_urgency_proxy_target(df)

    n = len(X)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]
    n_train = int(round(cfg.train_test_split * n))

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    feature_min = X_train.min(axis=0)
    feature_max = X_train.max(axis=0)
    span = np.where(feature_max - feature_min < 1e-8, 1.0, feature_max - feature_min)

    X_train_norm = (X_train - feature_min) / span
    X_test_norm = (X_test - feature_min) / span

    return HeartDiseaseData(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        X_train_norm=X_train_norm, X_test_norm=X_test_norm,
        feature_names=["age", "thallium_stress_test", "blood_pressure", "cholesterol", "chest_pain"],
        feature_min=feature_min, feature_max=feature_max,
    )


# --------------------------------------------------------------------------- #
def make_synthetic_fallback(n_samples: int = 300, seed: int = None) -> HeartDiseaseData:
   
    rng = np.random.default_rng(seed)
    age = rng.uniform(29, 77, n_samples)
    thal = rng.choice([3, 6, 7], n_samples).astype(float)
    trestbps = rng.uniform(94, 200, n_samples)
    chol = rng.uniform(126, 564, n_samples)
    cp = rng.choice([1, 2, 3, 4], n_samples).astype(float)
    num = rng.integers(0, 5, n_samples).astype(float)

    df = pd.DataFrame({
        "age": age, "thal": thal, "trestbps": trestbps,
        "chol": chol, "cp": cp, "num": num,
    })
    tmp_path = Path("/tmp/_hpp_fdrl_synthetic_heart.csv")
    df.to_csv(tmp_path, index=False)
    return load_and_prepare(str(tmp_path), seed=seed)


if __name__ == "__main__":
    try:
        data = load_and_prepare()
        print(f"Loaded REAL UCI data: {data.X_train.shape[0]} train / "
              f"{data.X_test.shape[0]} test samples")
    except FileNotFoundError as e:
        print(f"[warning] {e}\nFalling back to synthetic data for a smoke test only.")
        data = make_synthetic_fallback(seed=0)
        print(f"Synthetic fallback: {data.X_train.shape[0]} train / "
              f"{data.X_test.shape[0]} test samples")
