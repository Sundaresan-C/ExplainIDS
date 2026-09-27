"""Cleaning, splitting, scaling and class balancing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from config import FEATURES, SEED


class Preprocessor:
    """log1p compresses heavy-tailed rates/durations, then MinMax to [0, 1]."""

    def __init__(self):
        self.scaler = MinMaxScaler(clip=True)

    @staticmethod
    def _transform_raw(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.sign(X) * np.log1p(np.abs(X))

    def fit(self, X):
        self.scaler.fit(self._transform_raw(X))
        return self

    def transform(self, X) -> np.ndarray:
        return self.scaler.transform(self._transform_raw(X)).astype(np.float32)

    def inverse_transform(self, Z) -> np.ndarray:
        L = self.scaler.inverse_transform(Z)
        return np.sign(L) * np.expm1(np.abs(L))


def make_splits(df: pd.DataFrame, label_col="Label", test_size=0.2, val_size=0.1, balance=True, exclude=None):
    """Return dict with scaled train/val/test splits + fitted preprocessor/encoder.

    exclude: list of categories to drop from TRAIN/VAL only (used by the
    zero-day leave-one-family-out experiment). They remain in the test set.
    """
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    X, y = df[FEATURES], df[label_col].astype(str)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=val_size / (1 - test_size),
                                              stratify=y_tr, random_state=SEED)
    if exclude:
        keep_tr, keep_va = ~y_tr.isin(exclude), ~y_va.isin(exclude)
        X_tr, y_tr, X_va, y_va = X_tr[keep_tr], y_tr[keep_tr], X_va[keep_va], y_va[keep_va]

    prep = Preprocessor().fit(X_tr)
    enc = LabelEncoder().fit(y_tr)
    Xtr, Xva, Xte = prep.transform(X_tr), prep.transform(X_va), prep.transform(X_te)
    ytr, yva = enc.transform(y_tr), enc.transform(y_va)

    if balance:
        counts = np.bincount(ytr)
        k = int(max(1, min(5, counts.min() - 1)))
        if counts.min() < counts.max() and counts.min() > 1:
            # SMOTE on the training split only, never on val/test.
            target = {c: max(n, int(counts.max() * 0.5)) for c, n in enumerate(counts)}
            Xtr, ytr = SMOTE(sampling_strategy=target, k_neighbors=k, random_state=SEED).fit_resample(Xtr, ytr)

    return {
        "Xtr": Xtr, "ytr": ytr, "Xva": Xva, "yva": yva, "Xte": Xte,
        "yte_str": y_te.to_numpy(), "Xte_raw": X_te.reset_index(drop=True),
        "Xtr_raw_benign": X_tr[y_tr == "Benign"] if "Benign" in set(y_tr) else None,
        "prep": prep, "enc": enc,
    }
