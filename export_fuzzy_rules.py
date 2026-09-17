"""
export_fuzzy_rules.py
========================
"""

import argparse

import numpy as np
import pandas as pd

from fuzzy_system import TSFuzzySystem
from data_loader import load_and_prepare, make_synthetic_fallback


def export_rules(data_path: str = None, seed: int = 0,
                  ridge_alpha: float = 5.0, min_effective_weight: float = 3.0,
                  out_path: str = "fuzzy_rule_coefficients_324.csv") -> pd.DataFrame:
  
    try:
        data = load_and_prepare(data_path, seed=seed)
        data_source = "real UCI heart_disease.csv"
    except FileNotFoundError:
        data = make_synthetic_fallback(seed=seed)
        data_source = "SYNTHETIC fallback (real UCI file not found)"
    print(f"Data source: {data_source} "
          f"({len(data.X_train)} train / {len(data.X_test)} test patients)")

   
    fuzzy = TSFuzzySystem(seed=seed)
    fuzzy.fit(data.X_train, data.y_train,
              ridge_alpha=ridge_alpha, min_effective_weight=min_effective_weight)
    fuzzy.calibrate(data.X_train)

    assert len(fuzzy.rules) == 324, f"expected 324 rules, got {len(fuzzy.rules)}"

    
    rows = []
    for idx, rule in enumerate(fuzzy.rules):
        age_lbl, thal_lbl, bp_lbl, chol_lbl, cp_lbl = rule
        c0, c1, c2, c3, c4, c5 = fuzzy.consequent_coeffs[idx]
        rows.append({
            "rule_no": idx + 1,
            "age": age_lbl,
            "thallium_stress_test": thal_lbl,
            "blood_pressure": bp_lbl,
            "cholesterol": chol_lbl,
            "chest_pain": cp_lbl,
            "effective_training_weight": round(float(fuzzy.rule_effective_weight_[idx]), 4),
            "confidence": fuzzy.rule_confidence_[idx],
            "c0_intercept": c0,
            "c1_age": c1,
            "c2_thallium": c2,
            "c3_blood_pressure": c3,
            "c4_cholesterol": c4,
            "c5_chest_pain": c5,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    
    n_high_from_attr = fuzzy.n_rules_high_confidence_
    n_high_from_df = int((df["confidence"] == "high").sum())
    print(f"\nExported {len(df)} rules to {out_path}")
    print(f"  ridge_alpha={ridge_alpha}, min_effective_weight={min_effective_weight}")
    print(f"  high-confidence rules: {n_high_from_attr} (from fuzzy object) "
          f"== {n_high_from_df} (from exported CSV): "
          f"{'OK' if n_high_from_attr == n_high_from_df else 'MISMATCH — DO NOT TRUST THIS FILE'}")
    print(f"  low-confidence rules: {len(df) - n_high_from_df}")
    print(f"  rules with exactly zero effective weight: "
          f"{int((df['effective_training_weight'] == 0).sum())}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None,
                         help="Path to heart_disease.csv (default: CWD-independent "
                              "hpp_fdrl/data/heart_disease.csv, see data_loader.py)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=5.0,
                         help="MUST match whatever fuzzy_system.py's fit() uses "
                              "elsewhere in the codebase (training_loop.py, "
                              "run_comparison.py, ablation_*.py all use the "
                              "default 5.0 — only change this if you also change "
                              "it everywhere else).")
    parser.add_argument("--min-effective-weight", type=float, default=3.0)
    parser.add_argument("--out", default="fuzzy_rule_coefficients_324.csv")
    args = parser.parse_args()

    export_rules(args.data, args.seed, args.ridge_alpha,
                 args.min_effective_weight, args.out)
