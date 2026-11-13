"""Experiments.

1. Compression robustness: a head trained on clean features vs. one trained on clean
   plus compressed features, each tested at every compression level.
2. Calibration and abstention: ECE with and without temperature scaling, and a
   risk-coverage curve (balanced accuracy when abstaining on the least confident).

Splits are grouped by source video. Results are mean +/- std over folds and repeats.

  python -m v2.evaluate --features_dir feats --out_dir results --save_model results/model.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from v2.headmodel import EnsembleHead, pool_frames


def ece_score(p_fake: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error on the predicted class's confidence."""
    pred = (p_fake >= 0.5).astype(int)
    conf = np.maximum(p_fake, 1 - p_fake)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def safe_bacc(y, pred):
    return float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else float(np.mean(y == pred))


def load_levels(features_dir: Path):
    files = sorted(features_dir.glob("feats_*.npz"))
    if not files:
        raise SystemExit(f"No feats_*.npz files in {features_dir}")
    raw = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        raw.append({
            "level": f.stem.replace("feats_", ""),
            "severity": float(d["severity"]),
            "X": d["X"], "y": d["y"], "groups": d["groups"], "names": d["names"],
            "embedder": str(d["embedder"]), "detector": str(d["detector"]),
        })
    raw.sort(key=lambda r: r["severity"])
    common = set(raw[0]["names"].tolist())
    for r in raw[1:]:
        common &= set(r["names"].tolist())
    common = sorted(common)
    if len(common) < 20:
        raise SystemExit(f"Only {len(common)} videos are present at every level; too few to evaluate.")
    levels = []
    for r in raw:
        order = {n: i for i, n in enumerate(r["names"].tolist())}
        idx = [order[n] for n in common]
        levels.append({
            "level": r["level"], "severity": r["severity"],
            "Xp": pool_frames(r["X"][idx]), "y": r["y"][idx], "groups": r["groups"][idx],
            "embedder": r["embedder"], "detector": r["detector"],
        })
    return levels, common


def risk_coverage(p_fake, y, coverages):
    conf = np.maximum(p_fake, 1 - p_fake)
    order = np.argsort(-conf)
    out = []
    for c in coverages:
        k = max(2, int(round(c * len(y))))
        keep = order[:k]
        out.append(safe_bacc(y[keep], (p_fake[keep] >= 0.5).astype(int)))
    return out


def run(features_dir, out_dir, folds, seeds, members, C):
    out_dir.mkdir(parents=True, exist_ok=True)
    levels, names = load_levels(features_dir)
    y = levels[0]["y"]
    groups = levels[0]["groups"]
    n_real, n_fake = int((y == 0).sum()), int((y == 1).sum())
    print(f"videos={len(y)} real={n_real} fake={n_fake} groups={len(np.unique(groups))} "
          f"levels={[l['level'] for l in levels]} embedder={levels[0]['embedder']} "
          f"detector={levels[0]['detector']}")
    if levels[0]["embedder"].startswith("dummy"):
        print("WARNING: features come from the dummy test embedder; results are meaningless.")

    modes = {"clean-trained": [0], "compression-aug-trained": list(range(len(levels)))}
    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    metrics = {(m, l["level"]): {"auc": [], "bacc": [], "ece": [], "ece_raw": []}
               for m in modes for l in levels}
    pooled = {(m, l["level"]): {"p": [], "y": []} for m in modes for l in levels}

    for seed in range(seeds):
        skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed * 100 + fold)
            fit_rel, cal_rel = next(gss.split(tr, y[tr], groups[tr]))
            fit_idx, cal_idx = tr[fit_rel], tr[cal_rel]

            for mode, lv_idx in modes.items():
                Xf = np.concatenate([levels[i]["Xp"][fit_idx] for i in lv_idx])
                yf = np.concatenate([y[fit_idx] for _ in lv_idx])
                gf = np.concatenate([groups[fit_idx] for _ in lv_idx])
                Xc = np.concatenate([levels[i]["Xp"][cal_idx] for i in lv_idx])
                yc = np.concatenate([y[cal_idx] for _ in lv_idx])

                head = EnsembleHead(n_members=members, C=C, seed=seed * 1000 + fold)
                head.fit(Xf, yf, gf, Xc, yc)

                for l in levels:
                    pred = head.predict(l["Xp"][te])
                    yt = y[te]
                    key = (mode, l["level"])
                    metrics[key]["auc"].append(safe_auc(yt, pred["logit"]))
                    metrics[key]["bacc"].append(safe_bacc(yt, (pred["p_fake"] >= 0.5).astype(int)))
                    metrics[key]["ece"].append(ece_score(pred["p_fake"], yt))
                    metrics[key]["ece_raw"].append(ece_score(pred["p_fake_raw"], yt))
                    pooled[key]["p"].append(pred["p_fake"])
                    pooled[key]["y"].append(yt)
        print(f"  finished repeat {seed + 1}/{seeds}")

    def ms(v):
        v = np.array(v, dtype=float)
        v = v[~np.isnan(v)]
        return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))

    summary = {}
    for (mode, lvl), m in metrics.items():
        summary.setdefault(mode, {})[lvl] = {k: ms(v) for k, v in m.items()}
    rc = {}
    for (mode, lvl), d in pooled.items():
        rc.setdefault(mode, {})[lvl] = risk_coverage(np.concatenate(d["p"]), np.concatenate(d["y"]), coverages)

    (out_dir / "results.json").write_text(json.dumps({
        "n_videos": len(y), "n_real": n_real, "n_fake": n_fake,
        "levels": [{"level": l["level"], "severity": l["severity"]} for l in levels],
        "embedder": levels[0]["embedder"], "detector": levels[0]["detector"],
        "folds": folds, "repeats": seeds, "members": members, "C": C,
        "summary": summary, "risk_coverage": rc, "coverages": coverages,
    }, indent=2))

    lines = [f"Videos: {len(y)} (real {n_real}, fake {n_fake}); {folds}-fold grouped CV x {seeds} repeats; "
             f"embedder {levels[0]['embedder']}; detector {levels[0]['detector']}.\n",
             "| Training | Test level | AUC | Balanced acc | ECE (calibrated) | ECE (raw) |",
             "|---|---|---|---|---|---|"]
    for mode in modes:
        for l in levels:
            s = summary[mode][l["level"]]
            lines.append(f"| {mode} | {l['level']} | {s['auc'][0]:.3f} ± {s['auc'][1]:.3f} | "
                         f"{s['bacc'][0]:.3f} ± {s['bacc'][1]:.3f} | "
                         f"{s['ece'][0]:.3f} | {s['ece_raw'][0]:.3f} |")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    sev = [l["severity"] for l in levels]
    labels = [l["level"] for l in levels]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in [(axes[0], "auc", "AUC"), (axes[1], "bacc", "Balanced accuracy")]:
        for mode, color in [("clean-trained", "#c2410c"), ("compression-aug-trained", "#0f766e")]:
            mu = np.array([summary[mode][lv][key][0] for lv in labels])
            sd = np.array([summary[mode][lv][key][1] for lv in labels])
            ax.plot(sev, mu, marker="o", label=mode, color=color)
            ax.fill_between(sev, mu - sd, mu + sd, alpha=0.15, color=color)
        ax.set_xticks(sev)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_xlabel("Test-time compression (simulated)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs compression")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "compression.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    hardest = labels[-1]
    for mode, color in [("clean-trained", "#c2410c"), ("compression-aug-trained", "#0f766e")]:
        ax.plot(coverages, rc[mode][labels[0]], marker="o", color=color, label=f"{mode}, {labels[0]}")
        ax.plot(coverages, rc[mode][hardest], marker="s", linestyle="--", color=color, label=f"{mode}, {hardest}")
    ax.set_xlabel("Coverage (fraction of videos answered)")
    ax.set_ylabel("Balanced accuracy on answered videos")
    ax.set_title("Risk-coverage: abstain on least-confident")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "risk_coverage.png", dpi=150)
    plt.close(fig)
    return levels, y, groups


def save_final_model(levels, y, groups, path: Path, members: int, C: float):
    """Train on all videos at all levels (compression-augmented) and save for serving."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    fit_idx, cal_idx = next(gss.split(np.zeros(len(y)), y, groups))
    lv = range(len(levels))
    Xf = np.concatenate([levels[i]["Xp"][fit_idx] for i in lv])
    yf = np.concatenate([y[fit_idx] for _ in lv])
    gf = np.concatenate([groups[fit_idx] for _ in lv])
    Xc = np.concatenate([levels[i]["Xp"][cal_idx] for i in lv])
    yc = np.concatenate([y[cal_idx] for _ in lv])
    head = EnsembleHead(n_members=members, C=C, seed=7).fit(Xf, yf, gf, Xc, yc)
    joblib.dump({
        "head": head,
        "embedder": levels[0]["embedder"],
        "detector": levels[0]["detector"],
        "n_frames": 16,
        "sklearn_version": sklearn.__version__,
        "note": "Trained on DFDC sample features with simulated compression augmentation.",
    }, path)
    print(f"saved model to {path} (temperature={head.temperature_:.3f}, members={len(head.members)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("results"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--C", type=float, default=0.05, help="inverse L2 strength")
    ap.add_argument("--save_model", type=Path, default=None)
    args = ap.parse_args()

    levels, y, groups = run(args.features_dir, args.out_dir, args.folds, args.seeds, args.members, args.C)
    if args.save_model:
        save_final_model(levels, y, groups, args.save_model, args.members, args.C)


if __name__ == "__main__":
    main()
