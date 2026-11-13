"""Manifest -> per-frame embeddings, optionally after simulated compression.

One run writes feats_<level>.npz with:
  X (V, n_frames, D), y (V,), groups (V,), names (V,), severity, embedder, detector

With --crf / --scale each video is re-encoded with ffmpeg (libx264) before face
extraction, so crops come from the degraded frames. This approximates, but is not
identical to, platform re-encoding.

  python -m v2.extract --manifest manifest.csv --out feats --level orig
  python -m v2.extract --manifest manifest.csv --out feats --level crf35 --crf 35 --severity 35
  python -m v2.extract --manifest manifest.csv --out feats --level social --scale 480 --crf 32 --severity 38
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from v2.faces import FaceCropper, sample_face_crops


def reencode(src: Path, dst: Path, crf: int | None, scale: int | None) -> bool:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an"]
    if scale:
        cmd += ["-vf", f"scale=-2:{scale}"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf if crf is not None else 23), str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return r.returncode == 0 and dst.exists()


def read_manifest(path: Path, limit: int | None):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((Path(r["path"]), int(r["label"]), r["group"]))
    if limit:
        # keep both classes when limiting
        real = [r for r in rows if r[1] == 0]
        fake = [r for r in rows if r[1] == 1]
        k = max(1, limit // 2)
        rows = real[:k] + fake[:limit - k]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--level", type=str, default="orig")
    ap.add_argument("--crf", type=int, default=None)
    ap.add_argument("--scale", type=int, default=None, help="downscale to this height before encoding")
    ap.add_argument("--severity", type=float, default=None, help="x-axis value for plots (default: crf or 0)")
    ap.add_argument("--n_frames", type=int, default=16)
    ap.add_argument("--embedder", choices=["clip", "dummy"], default="clip")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from v2.embed import get_embedder

    args.out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest, args.limit)
    severity = args.severity if args.severity is not None else float(args.crf or 0)
    degrade = args.crf is not None or args.scale is not None

    cropper = FaceCropper()
    local = threading.local()

    def thread_cropper() -> FaceCropper:
        # One detector per worker thread; a shared OpenCV DNN detector is not thread-safe.
        if not hasattr(local, "c"):
            local.c = FaceCropper()
        return local.c

    print(f"[extract] level={args.level} videos={len(rows)} detector={cropper.name} "
          f"embedder={args.embedder} degrade={degrade}")
    embedder = get_embedder(args.embedder)
    tmp_root = Path(tempfile.mkdtemp(prefix="dfx_"))

    def prepare(item):
        path, label, group = item
        src = path
        if degrade:
            dst = tmp_root / f"{path.stem}_{args.level}.mp4"
            if not reencode(path, dst, args.crf, args.scale):
                return item, None, 0
            src = dst
        crops, found = sample_face_crops(src, thread_cropper(), n_frames=args.n_frames)
        if degrade:
            Path(src).unlink(missing_ok=True)
        return item, crops, found

    X, y, groups, names = [], [], [], []
    skipped = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (item, crops, found) in enumerate(pool.map(prepare, rows), 1):
            path, label, group = item
            if crops is None:
                skipped += 1
            else:
                X.append(embedder(crops))
                y.append(label)
                groups.append(group)
                names.append(path.name)
            if i % 20 == 0 or i == len(rows):
                rate = (time.time() - t0) / i
                print(f"[extract] {i}/{len(rows)} kept={len(X)} skipped={skipped} "
                      f"({rate:.1f}s/video, ~{rate * (len(rows) - i) / 60:.1f} min left)")

    if not X:
        raise SystemExit("No videos produced features. Check face detection and paths.")

    out_file = args.out / f"feats_{args.level}.npz"
    np.savez_compressed(
        out_file,
        X=np.stack(X).astype(np.float32),
        y=np.array(y, dtype=np.int64),
        groups=np.array(groups),
        names=np.array(names),
        severity=np.float32(severity),
        embedder=embedder.name,
        detector=cropper.name,
    )
    print(f"[extract] saved {out_file} (videos kept={len(X)}, skipped={skipped})")


if __name__ == "__main__":
    main()
