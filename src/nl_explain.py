"""Turn SHAP attributions into a plain-English explanation (our extension)."""
from __future__ import annotations

import numpy as np

FRIENDLY = {
    "Protocol": "the transport protocol",
    "Flow Duration": "the flow duration",
    "Total Fwd Packets": "the number of packets sent by the initiator",
    "Total Bwd Packets": "the number of reply packets",
    "Total Length of Fwd Packets": "the bytes sent by the initiator",
    "Total Length of Bwd Packets": "the bytes sent back",
    "Flow Bytes/s": "the byte rate",
    "Flow Packets/s": "the packet rate",
    "Fwd Packets/s": "the initiator's packet rate",
    "Bwd Packets/s": "the reply packet rate",
    "Flow IAT Mean": "the average gap between packets",
    "Flow IAT Min": "the smallest gap between packets",
    "Flow IAT Max": "the largest gap between packets",
    "SYN Flag Count": "the number of SYN (connection-open) flags",
    "RST Flag Count": "the number of RST (connection-reset) flags",
    "FIN Flag Count": "the number of FIN (connection-close) flags",
    "ACK Flag Count": "the number of ACK flags",
    "PSH Flag Count": "the number of PSH (push data) flags",
    "URG Flag Count": "the number of URG flags",
    "Down/Up Ratio": "the reply-to-request packet ratio",
    "Average Packet Size": "the average packet size",
    "Init Fwd Win Bytes": "the initiator's initial TCP window",
    "Init Bwd Win Bytes": "the responder's initial TCP window",
    "Packet Length Mean": "the mean payload length",
    "Packet Length Max": "the largest payload",
    "Fwd Packet Length Mean": "the initiator's mean payload size",
    "Bwd Packet Length Mean": "the responder's mean payload size",
    "Bwd IAT Mean": "the average gap between reply packets",
    "Bwd IAT Std": "the variation in reply-packet timing",
    "Bwd IAT Min": "the smallest gap between reply packets",
    "Bwd IAT Max": "the largest gap between reply packets",
    "Fwd IAT Mean": "the average gap between the initiator's packets",
    "Fwd IAT Std": "the variation in the initiator's packet timing",
    "Fwd IAT Min": "the smallest gap between the initiator's packets",
    "Flow IAT Std": "the variation in packet timing",
    "Fwd Header Length": "the initiator's header bytes",
    "Bwd Header Length": "the reply header bytes",
    "Packet Length Std": "the variation in payload size",
    "Fwd Packet Length Std": "the variation in the initiator's payload size",
    "Fwd Packet Length Max": "the initiator's largest payload",
    "Bwd Packet Length Max": "the responder's largest payload",
}

_PROTO = {6: "TCP", 17: "UDP", 1: "ICMP", 47: "GRE", 58: "ICMPv6"}


def _fmt(v: float) -> str:
    v = float(v)
    if abs(v) >= 1e6:
        return f"{v / 1e6:,.1f}M"
    if abs(v) >= 1e4:
        return f"{v / 1e3:,.1f}k"
    if abs(v) >= 100 or v == int(v):
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def explain_text(pred: str, confidence: float, features, raw_values, shap_values, benign_median: dict,
                 top_k: int = 3, verdict: str | None = None, anomaly: float | None = None,
                 threshold: float | None = None) -> str:
    raw_values, shap_values = np.asarray(raw_values, float), np.asarray(shap_values, float)
    order = np.argsort(-shap_values)            # features that pushed TOWARD the predicted class
    reasons = []
    # Walk the features that pushed toward the prediction (highest SHAP first) and keep only
    # those that actually differ from normal traffic - "close to normal" is not an explanation.
    for i in order[:12]:
        if shap_values[i] <= 0 or len(reasons) >= top_k:
            break
        f, v = features[i], raw_values[i]
        name = FRIENDLY.get(f, f"'{f}'")
        if f == "Protocol":
            reasons.append(f"it uses {_PROTO.get(int(v), f'protocol {int(v)}')}")
            continue
        ref = benign_median.get(f)
        if ref is None:
            reasons.append(f"{name} is {_fmt(v)}")
            continue
        if v > ref * 1.5 + 1e-9:
            rel = "much higher than" if v > ref * 5 + 1 else "higher than"
        elif v < ref / 1.5 - 1e-9:
            rel = "much lower than" if v < ref / 5 else "lower than"
        else:
            continue
        reasons.append(f"{name} is {_fmt(v)} ({rel} normal traffic, typically {_fmt(ref)})")

    head = verdict or pred
    if head == "Unknown attack":
        s = (f"Flagged as an unknown (zero-day) attack. The classifier's best guess was "
             f"'{pred}' ({confidence:.0%}), but the flow does not look like normal traffic: its anomaly score "
             f"is {anomaly:.4f}, above the benign threshold of {threshold:.4f}.")
        if reasons:
            s += " Most unusual aspects: " + "; ".join(reasons) + "."
        return s
    if head == "Benign":
        return f"Classified as Benign ({confidence:.0%} confidence). The traffic pattern matches normal device behaviour."
    s = f"Classified as {head} ({confidence:.0%} confidence)"
    s += (", mainly because " + "; ".join(reasons) + ".") if reasons else "."
    return s
