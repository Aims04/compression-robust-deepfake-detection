"""Build a manifest CSV (path,label,group) from DFDC's metadata.json.

label: 1 = fake, 0 = real.
group: the original real video a clip derives from. Real clips are their own
group. Splitting by group keeps each source video, and all fakes made from it,
on one side of a train/test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfdc_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("manifest.csv"))
    args = ap.parse_args()

    meta = json.loads((args.dfdc_dir / "metadata.json").read_text())
    n_real = n_fake = n_missing = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "group"])
        for name, info in meta.items():
            p = args.dfdc_dir / name
            if not p.exists():
                n_missing += 1
                continue
            is_fake = str(info.get("label", "")).upper() == "FAKE"
            group = info.get("original") or name
            w.writerow([str(p), int(is_fake), group])
            n_fake += is_fake
            n_real += not is_fake
    print(f"wrote {args.out}: real={n_real} fake={n_fake} missing_files={n_missing}")


if __name__ == "__main__":
    main()
