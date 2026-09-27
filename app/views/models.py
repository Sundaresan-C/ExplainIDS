import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import T, chart, figure, header, is_synthetic, load_json, note, section

header("Model comparison", "Classical baselines, the hybrid CNN → XGBoost and its fusion variant, all evaluated on the "
       "same held-out test set.")
m = load_json("metrics.json")
if not m:
    st.error("Run `python src/train.py` first."); st.stop()
if is_synthetic():
    note("Synthetic smoke-test data. Retrain on CIC IoT-DIAD 2024 for the real numbers.")

df = pd.DataFrame(m["models"]).T.astype(float)
df.index.name = "Model"
tbl = df[["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr", "train_time_s", "inference_ms_per_flow"]].copy()
tbl["inference_ms_per_flow"] *= 1000
tbl = tbl.rename(columns={"accuracy": "Accuracy", "precision_macro": "Precision", "recall_macro": "Recall", "f1_macro": "Macro F1",
                          "roc_auc_ovr": "ROC-AUC", "train_time_s": "Train (s)", "inference_ms_per_flow": "µs / flow"})
pc = {c: st.column_config.NumberColumn(c, format="percent") for c in ["Accuracy", "Precision", "Recall", "Macro F1"]}
st.dataframe(tbl.reset_index(), width="stretch", hide_index=True,
             column_config=pc | {"ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
                                 "Train (s)": st.column_config.NumberColumn(format="%.1f"),
                                 "µs / flow": st.column_config.NumberColumn(format="%.1f")})

hl = {"Hybrid CNN-XGBoost", "Fusion CNN-XGBoost"}
d = df.sort_values("f1_macro")
fig = go.Figure(go.Bar(x=d["f1_macro"], y=d.index, orientation="h",
                       marker_color=[T()["accent"] if k in hl else T()["neutral"] for k in d.index],
                       text=[f"{v:.2%}" for v in d["f1_macro"]], textposition="outside"))
lo = max(0, d["f1_macro"].min() - .05)
chart(fig, 330, title="Macro F1 by model", xaxis=dict(range=[lo, 1.0], tickformat=".0%"), bargap=.35)

st.caption("Evaluation protocol: identifier features (IPs, ports, timestamps) are excluded, metrics are macro-averaged "
           "over 8 classes, and SMOTE is applied to the training split only.")

section("Per-class performance")
a, b = st.columns([1, 1], gap="medium")
with a:
    figure("confusion_matrix_deployed.png")
with b:
    rc = pd.DataFrame(m["per_class_report"]).T
    rc = rc[~rc.index.isin(["accuracy", "macro avg", "weighted avg"])][["precision", "recall", "f1-score", "support"]]
    st.dataframe(rc.reset_index(names="Class"), width="stretch", hide_index=True,
                 column_config={k: st.column_config.ProgressColumn(k.capitalize(), format="percent", min_value=0, max_value=1)
                                for k in ["precision", "recall", "f1-score"]} | {"support": st.column_config.NumberColumn("Support", format="%d")})

section("Explainability (global)")
c1, c2 = st.columns(2, gap="medium")
with c1:
    figure("shap_hybrid_bar.png", "Global SHAP importance: attack score")
with c2:
    figure("shap_per_class_heatmap.png", "Per-class TreeSHAP importance")
with st.expander("Training curves and SHAP beeswarm"):
    e1, e2 = st.columns(2)
    with e1:
        figure("cnn_training_curves.png")
    with e2:
        figure("shap_hybrid_beeswarm.png")
