"""Out-of-fold predictions per video, for choosing demo clips.

Each video is scored by a head trained on folds that exclude its source group, so
the prediction is not in-sample. Writes a CSV and prints a short list of clips:
confident correct fakes and reals, an uncertain case, and a confident error.

  python -m v2.pick_demo_clips --features feats/feats_orig.npz --out oof_predictions.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from v2.headmodel import EnsembleHead, pool_frames


def out_of_fold(X, y, groups, seed: int, n_splits: int = 5):
    Xp = pool_frames(X)
    p = np.full(len(y), np.nan)
    rng = np.random.RandomState(seed)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in splitter.split(Xp, y, groups):
        train_groups = np.unique(groups[train])
        cal_groups = rng.choice(train_groups, size=max(1, len(train_groups) // 5), replace=False)
        is_cal = np.isin(groups[train], cal_groups)
        fit, cal = train[~is_cal], train[is_cal]
        head = EnsembleHead(seed=seed).fit(Xp[fit], y[fit], groups[fit], Xp[cal], y[cal])
        p[test] = head.predict(Xp[test])["p_fake"]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="feats_orig.npz")
    ap.add_argument("--out", default="oof_predictions.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--abstain", type=float, default=0.65)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, y, groups, names = d["X"], d["y"], d["groups"], d["names"]
    p = out_of_fold(X, y, groups, args.seed)

    conf = np.maximum(p, 1 - p)
    verdict = np.where(conf < args.abstain, "uncertain", np.where(p >= 0.5, "fake", "real"))
    truth = np.where(y == 1, "fake", "real")
    df = pd.DataFrame({"video": names, "truth": truth, "p_fake": p.round(3),
                       "verdict": verdict, "confidence": conf.round(3)})
    df.to_csv(args.out, index=False)

    def top(mask, n, by="confidence", asc=False):
        return df[mask].sort_values(by, ascending=asc).head(n)

    print(f"overall accuracy when not abstaining: "
          f"{(df[df.verdict != 'uncertain'].verdict == df[df.verdict != 'uncertain'].truth).mean():.3f}")
    print("\nconfident, correct fakes:")
    print(top((df.truth == "fake") & (df.verdict == "fake"), 3).to_string(index=False))
    print("\nconfident, correct reals:")
    print(top((df.truth == "real") & (df.verdict == "real"), 3).to_string(index=False))
    print("\nuncertain:")
    print(top(df.verdict == "uncertain", 2, by="confidence", asc=True).to_string(index=False))
    print("\nconfident errors:")
    print(top((df.verdict != "uncertain") & (df.verdict != df.truth), 2).to_string(index=False))


if __name__ == "__main__":
    main()
