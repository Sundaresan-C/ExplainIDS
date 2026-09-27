"""Inference engine used by the web app: classify, detect zero-days, explain."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from config import FEATURES, MITIGATION, MODELS, SEVERITY
from data_loader import to_feature_frame
from models import FlowAutoencoder, HybridCNNXGB, load
from nl_explain import explain_text

CONF_THRESHOLD = 0.60   # below this AND anomalous -> "Unknown attack"


class Detector:
    def __init__(self, folder=MODELS):
        self.folder = folder
        cfg = json.loads((folder / "deploy.json").read_text())
        self.cfg = cfg
        self.prep = load(folder / "preprocessor.joblib")
        self.enc = load(folder / "label_encoder.joblib")
        self.model = HybridCNNXGB.load(folder, cfg["model_name"])
        self.ae = FlowAutoencoder.load(folder)
        self.background = np.load(folder / "shap_background.npy")
        self.benign_median = json.loads((folder / "benign_median.json").read_text())
        self.classes = list(self.enc.classes_)

    # ------------------------------------------------------------------
    def analyse(self, df: pd.DataFrame, extras: bool = False):
        """df: any frame containing CICFlowMeter columns (+ optional metadata).

        Returns the results table; with extras=True also a dict holding the scaled
        features "Z", raw features "X_raw", class probabilities "proba" and latency.
        """
        X_raw = to_feature_frame(df)
        Z = self.prep.transform(X_raw)
        t = time.perf_counter()
        P = self.model.predict_proba(Z)
        ae = self.ae.score(Z)
        latency_ms = (time.perf_counter() - t) * 1000 / max(len(Z), 1)
        pred = np.array(self.classes)[P.argmax(1)]
        conf = P.max(1)
        anomalous = ae > self.ae.threshold
        unknown = anomalous & ((pred == "Benign") | (conf < CONF_THRESHOLD))
        verdict = np.where(unknown, "Unknown attack", pred)

        out = pd.DataFrame(index=df.index)
        for c in ("Src IP", "Src Port", "Dst IP", "Dst Port"):
            if c in df.columns:
                out[c] = df[c].values
        out["Protocol"] = X_raw["Protocol"].astype(int).values
        out["Packets"] = (X_raw["Total Fwd Packets"] + X_raw["Total Bwd Packets"]).astype(int).values
        out["Predicted"] = pred
        out["Confidence"] = conf.round(4)
        out["Anomaly Score"] = ae.round(5)
        out["Verdict"] = verdict
        out["Severity"] = [SEVERITY.get(v, "High") for v in verdict]
        if extras:
            return out, {"Z": Z, "X_raw": X_raw.reset_index(drop=True), "proba": P, "latency_ms": latency_ms}
        return out

    # ------------------------------------------------------------------
    def shap_for(self, z_row: np.ndarray, class_idx: int, max_evals: int | None = None):
        """Model-agnostic SHAP (permutation) for one flow, for one class."""
        import shap
        f = lambda X: self.model.predict_proba(np.asarray(X, dtype=np.float32))[:, class_idx]
        masker = shap.maskers.Independent(self.background, max_samples=len(self.background))
        expl = shap.PermutationExplainer(f, masker, seed=0)
        sv = expl(z_row.reshape(1, -1), max_evals=max_evals or 2 * len(FEATURES) + 1, silent=True)
        return sv.values[0], float(sv.base_values[0])

    def explain(self, z: np.ndarray, raw: np.ndarray, P: np.ndarray, verdict: str, anomaly: float):
        """Explain one analysed flow (z = scaled row, raw = original feature values, P = class probabilities)."""
        cidx = int(P.argmax())
        values, base = self.shap_for(z, cidx)
        text = explain_text(self.classes[cidx], float(P[cidx]), FEATURES, raw, values, self.benign_median,
                            verdict=verdict, anomaly=anomaly, threshold=self.ae.threshold)
        return {"shap": values, "base": base, "raw": raw, "class": self.classes[cidx], "proba": P,
                "text": text, "mitigation": MITIGATION.get(verdict, MITIGATION["Unknown attack"]),
                "verdict": verdict}
