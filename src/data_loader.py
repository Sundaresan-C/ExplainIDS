"""Load CIC IoT-DIAD 2024 (or any CICFlowMeter CSV dataset) into one tidy table.

Usage:
    python src/data_loader.py                    # reads data/raw/**.csv
    python src/data_loader.py --source sample    # reads data/sample/**.csv
    python src/data_loader.py --per-class 30000  # cap rows per category

Labels are taken from a 'Label' column when it is informative; otherwise they
are inferred from the folder / file name (the DIAD release organises CSVs by
attack category and attack name).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (ALIAS_LOOKUP, CATEGORIES, DATA_PROCESSED, DATA_RAW, DATA_SAMPLE,
                    FEATURES, SEED, _key, to_category)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any CICFlowMeter column spelling to our canonical names."""
    ren = {}
    for c in df.columns:
        k = _key(c)
        if k in ALIAS_LOOKUP:
            ren[c] = ALIAS_LOOKUP[k]
        elif k == "label":
            ren[c] = "Label"
    df = df.rename(columns=ren)
    return df.loc[:, ~df.columns.duplicated()]


def to_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns, add missing features as 0, coerce to float, clean inf/NaN."""
    df = normalise_columns(df)
    out = pd.DataFrame(index=df.index)
    for f in FEATURES:
        out[f] = pd.to_numeric(df[f], errors="coerce") if f in df.columns else 0.0
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    return out


def _label_for(df: pd.DataFrame, path: Path) -> pd.Series:
    path_cat = to_category(str(path))
    if "Label" in df.columns:
        lab = df["Label"].astype(str).map(to_category)
        # Uninformative label column (e.g. "NeedManualLabel") -> use the path.
        if lab.notna().mean() > 0.5:
            return lab.fillna(path_cat) if path_cat else lab
    return pd.Series(path_cat, index=df.index)


def load_dataset(source: str = "raw", per_class: int = 25_000, chunksize: int = 200_000) -> pd.DataFrame:
    base = DATA_RAW if source == "raw" else DATA_SAMPLE
    files = sorted(base.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {base}. Download CIC IoT-DIAD 2024 flow CSVs "
            f"(AD_Flow-based-features) into data/raw/ or run src/make_sample_data.py.")
    rng = np.random.default_rng(SEED)
    buckets = {c: [] for c in CATEGORIES}
    counts = {c: 0 for c in CATEGORIES}
    for fp in files:
        print(f"reading {fp.relative_to(base)}")
        for chunk in pd.read_csv(fp, chunksize=chunksize, low_memory=False, on_bad_lines="skip"):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            labels = _label_for(normalise_columns(chunk.copy()), fp.relative_to(base))
            feats = to_feature_frame(chunk)
            feats["Label"] = labels.values
            feats = feats.dropna(subset=["Label"])
            for cat, g in feats.groupby("Label"):
                room = per_class * 3 - counts[cat]   # oversample then subsample
                if room <= 0:
                    continue
                if len(g) > room:
                    g = g.iloc[rng.choice(len(g), room, replace=False)]
                buckets[cat].append(g)
                counts[cat] += len(g)
    parts = []
    for cat, frames in buckets.items():
        if not frames:
            continue
        g = pd.concat(frames, ignore_index=True)
        if len(g) > per_class:
            g = g.sample(per_class, random_state=SEED)
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"dropped {before - len(df)} duplicate rows")
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["raw", "sample"], default="raw")
    ap.add_argument("--per-class", type=int, default=25_000)
    args = ap.parse_args()
    df = load_dataset(args.source, args.per_class)
    out = DATA_PROCESSED / "flows.parquet"
    df.to_parquet(out, index=False)
    name = "CIC IoT-DIAD 2024 (flow-based features)" if args.source == "raw" else "Synthetic sample (smoke test only)"
    (DATA_PROCESSED / "source.json").write_text(json.dumps({"source": name, "per_class": args.per_class}))
    print(df["Label"].value_counts())
    print(f"saved {len(df)} rows x {df.shape[1]} cols -> {out}")


if __name__ == "__main__":
    main()
