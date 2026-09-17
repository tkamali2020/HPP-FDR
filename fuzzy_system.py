"""
fuzzy_system.py
===============
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from config import CONFIG


# --------------------------------------------------------------------------- #
# Triangular membership function
# --------------------------------------------------------------------------- #
def triangular_mf(x: float, a: float, b: float, c: float) -> float:
   
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


@dataclass
class InputVariable:
    name: str
    mf_labels: List[str]

    mf_params: List[Tuple[float, float, float]]

    def memberships(self, x: float) -> Dict[str, float]:
        return {label: triangular_mf(x, *params)
                for label, params in zip(self.mf_labels, self.mf_params)}


def build_default_input_variables() -> List[InputVariable]:
    age = InputVariable(
        name="age",
        mf_labels=["young", "middle_aged", "elderly"],
        mf_params=[(20, 30, 45), (35, 50, 65), (55, 70, 90)],
    )
 
    stress_test = InputVariable(
        name="thallium_stress_test",
        mf_labels=["normal", "fixed_defect", "reversible_defect"],
        mf_params=[(2.0, 3.0, 4.5), (4.5, 6.0, 6.5), (6.5, 7.0, 8.0)],
    )
    blood_pressure = InputVariable(
        name="blood_pressure",
        mf_labels=["low", "medium", "high"],
        mf_params=[(80, 100, 120), (110, 130, 150), (140, 160, 220)],
    )
  
    cholesterol = InputVariable(
        name="cholesterol",
        mf_labels=["low", "medium", "high"],
        mf_params=[(100, 150, 200), (180, 220, 260), (240, 320, 600)],
    )

    chest_pain = InputVariable(
        name="chest_pain",
        mf_labels=["typical_angina", "atypical_angina", "non_anginal", "asymptomatic"],
        mf_params=[(0.5, 1.0, 1.5), (1.5, 2.0, 2.5), (2.5, 3.0, 3.5), (3.5, 4.0, 4.5)],
    )
    return [age, stress_test, blood_pressure, cholesterol, chest_pain]


# --------------------------------------------------------------------------- #

def enumerate_rules(input_vars: List[InputVariable]) -> List[Tuple[str, ...]]:
    from itertools import product
    label_lists = [v.mf_labels for v in input_vars]
    return list(product(*label_lists))


class TSFuzzySystem:
    def __init__(self, input_vars: List[InputVariable] = None, seed: int = None):
        self.input_vars = input_vars or build_default_input_variables()
        self.rules = enumerate_rules(self.input_vars)  # 324 tuples of labels
        assert len(self.rules) == CONFIG.fuzzy.n_rules, \
            f"expected {CONFIG.fuzzy.n_rules} rules, got {len(self.rules)}"

      
        rng = np.random.default_rng(seed)
        n_inputs = len(self.input_vars)
        self.consequent_coeffs = rng.normal(0, 0.05, size=(len(self.rules), n_inputs + 1))

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray, ridge_alpha: float = 5.0,
            min_effective_weight: float = 3.0):
        
        self._check_scale_sanity(X)

        n_inputs = len(self.input_vars)
        n_samples = len(X)

    
        self._feat_mean = X.mean(axis=0)
        self._feat_std = np.where(X.std(axis=0) < 1e-8, 1.0, X.std(axis=0))
        X_std = (X - self._feat_mean) / self._feat_std

        self.rule_effective_weight_ = np.zeros(len(self.rules))
        self.rule_confidence_ = np.array(["low"] * len(self.rules), dtype=object)

        for r_idx, rule in enumerate(self.rules):
            w = self._rule_weights_batch(X, rule)          
            eff_weight = float(w.sum())
            self.rule_effective_weight_[r_idx] = eff_weight
            self.rule_confidence_[r_idx] = (
                "high" if eff_weight >= min_effective_weight else "low"
            )

            X_ext = np.hstack([np.ones((n_samples, 1)), X_std])   
            W = w  
            WX = X_ext * W[:, None]
            
         
            penalty = ridge_alpha * np.eye(X_ext.shape[1])
            penalty[0, 0] = 1e-8
            A = X_ext.T @ WX + penalty
            b = X_ext.T @ (W * y)
            coeffs_std = np.linalg.solve(A, b)  

         
            c0_std, c_rest_std = coeffs_std[0], coeffs_std[1:]
            c_rest_raw = c_rest_std / self._feat_std
            c0_raw = c0_std - float(np.sum(c_rest_std * self._feat_mean / self._feat_std))
            self.consequent_coeffs[r_idx] = np.concatenate([[c0_raw], c_rest_raw])

        self.n_rules_fitted_ = len(self.rules)  
        self.n_rules_high_confidence_ = int((self.rule_confidence_ == "high").sum())

    def _check_scale_sanity(self, X: np.ndarray):
      
        max_mf_support = max(
            foot for var in self.input_vars for params in var.mf_params for foot in params
        )
        if X.size > 0 and float(np.max(np.abs(X))) <= 1.5 and max_mf_support > 5.0:
            raise ValueError(
                f"TSFuzzySystem.fit()/priority_score(): input data appears "
                f"to be normalized to [0,1] (max abs value = "
                f"{float(np.max(np.abs(X))):.3f}), but membership functions "
                f"are defined on raw clinical units (max MF support = "
                f"{max_mf_support:.1f}). Pass RAW clinical rows (e.g. "
                f"heart_data.X_train), not heart_data.X_train_norm."
            )

    # ------------------------------------------------------------------ #
    def _input_memberships(self, x: np.ndarray) -> List[Dict[str, float]]:
        return [var.memberships(x[i]) for i, var in enumerate(self.input_vars)]

    def _rule_weight(self, memberships: List[Dict[str, float]], rule: Tuple[str, ...]) -> float:
        
        w = 1.0
        for var_memberships, label in zip(memberships, rule):
            w *= var_memberships.get(label, 0.0)
        return w

    def _rule_weights_batch(self, X: np.ndarray, rule: Tuple[str, ...]) -> np.ndarray:
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            memberships = self._input_memberships(x)
            out[i] = self._rule_weight(memberships, rule)
        return out

    # ------------------------------------------------------------------ #
    def priority_score(self, x: np.ndarray) -> float:
        
        memberships = self._input_memberships(x)
        weights = np.array([self._rule_weight(memberships, r) for r in self.rules])
        x_ext = np.concatenate([[1.0], x])
        rule_outputs = self.consequent_coeffs @ x_ext  

        total_weight = weights.sum()
        if total_weight < 1e-8:
            return 0.0
        return float((weights @ rule_outputs) / total_weight)

    # ------------------------------------------------------------------ #
    def normalized_priority(self, x: np.ndarray, score_min: float, score_max: float) -> float:
      
        if score_max <= score_min:
            return 0.5
        p = (self.priority_score(x) - score_min) / (score_max - score_min)
        return float(min(max(p, 0.0), 1.0))

    # ------------------------------------------------------------------ #
    def calibrate(self, X_raw: np.ndarray):
     
        scores = np.array([self.priority_score(x) for x in X_raw])
        self.score_min = float(scores.min())
        self.score_max = float(scores.max())

    def normalized_priority_calibrated(self, x_raw: np.ndarray) -> float:
       
        if not hasattr(self, "score_min"):
            raise RuntimeError(
                "TSFuzzySystem.calibrate(X_raw) must be called once (after "
                "fit()) before normalized_priority_calibrated()."
            )
        return self.normalized_priority(x_raw, self.score_min, self.score_max)

    # ------------------------------------------------------------------ #
    def discretize_label(self, p_t_normalized: float) -> str:
        
        b1, b2 = CONFIG.fuzzy.discretization_bounds
        if p_t_normalized < b1:
            return CONFIG.fuzzy.priority_labels[0]
        elif p_t_normalized < b2:
            return CONFIG.fuzzy.priority_labels[1]
        return CONFIG.fuzzy.priority_labels[2]
