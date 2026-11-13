"""Classification head: pooled frame embeddings -> calibrated fake probability.

Frame embeddings (n_frames, D) from a frozen backbone are pooled to [mean, std]
over frames. A bagged ensemble of L2-regularised logistic regressions (group
bootstrap) produces a logit, temperature scaling calibrates it, and the spread of
member logits serves as a second uncertainty signal.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def pool_frames(X: np.ndarray) -> np.ndarray:
    """(V, n, D) -> (V, 2D): mean and standard deviation over frames."""
    return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T > 0 minimising the NLL of sigmoid(logits / T)."""
    if len(np.unique(y)) < 2 or len(y) < 4:
        return 1.0

    def nll(log_t: float) -> float:
        p = np.clip(sigmoid(logits / np.exp(log_t)), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


class EnsembleHead:
    def __init__(self, n_members: int = 5, C: float = 0.05, seed: int = 0):
        self.n_members, self.C, self.seed = n_members, C, seed
        self.members: list = []
        self.temperature_: float = 1.0

    def fit(self, Xp, y, groups, Xp_cal=None, y_cal=None):
        rng = np.random.RandomState(self.seed)
        y = np.asarray(y)
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        self.members = []
        attempts = 0
        while len(self.members) < self.n_members and attempts < self.n_members * 20:
            attempts += 1
            chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([np.where(groups == g)[0] for g in chosen])
            if len(np.unique(y[idx])) < 2:
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced"),
            )
            clf.fit(Xp[idx], y[idx])
            self.members.append(clf)
        if not self.members:
            raise RuntimeError("No ensemble member could be trained; check class balance.")
        if Xp_cal is not None and y_cal is not None:
            mean_logit, _ = self.raw_logits(Xp_cal)
            self.temperature_ = fit_temperature(mean_logit, np.asarray(y_cal))
        return self

    def raw_logits(self, Xp):
        """Uncalibrated mean logit and member disagreement (std of member logits)."""
        member_logits = np.stack([m.decision_function(Xp) for m in self.members], axis=0)
        return member_logits.mean(axis=0), member_logits.std(axis=0)

    def predict(self, Xp):
        logit, disagreement = self.raw_logits(Xp)
        return {
            "p_fake": sigmoid(logit / self.temperature_),
            "p_fake_raw": sigmoid(logit),
            "logit": logit,
            "disagreement": disagreement,
        }
