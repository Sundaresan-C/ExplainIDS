import streamlit as st

from common import deployed_metrics, figure, header, is_synthetic, kpis, load_json, note, section, theme

header("ExplainIDS", "Explainable, zero-day-aware network intrusion detection built on a hybrid CNN → XGBoost model.")

m = load_json("metrics.json")
if m:
    d = deployed_metrics(m)
    kpis([
        ("Accuracy", f"{d['accuracy']:.2%}", "held-out test set"),
        ("Macro F1", f"{d['f1_macro']:.2%}", f"{len(m['dataset']['class_counts'])} classes"),
        ("Attack vs benign AUC", f"{m['binary_attack_vs_benign']['roc_auc']:.3f}", "binary view"),
        ("Latency", f"{d['inference_ms_per_flow'] * 1000:.0f} µs", "per flow, CPU"),
    ])
    if is_synthetic():
        note("These numbers come from the <b>synthetic smoke-test sample</b>. To get the real results, put the CIC IoT-DIAD 2024 "
             "CSVs in <code>data/raw/</code> and run <code>run_pipeline.bat</code>.")
else:
    st.info("Train the models first: `python src/train.py`.")

section("Architecture")
figure("architecture_dark.png" if theme() == "dark" else "architecture_app.png")

c1, c2 = st.columns(2, gap="medium")
with c1:
    st.markdown("""<div class="card"><h4>How it works</h4>
<p>A 1D-CNN learns deep features from network-flow statistics and XGBoost classifies them into benign traffic or one of
seven attack categories. A benign-only autoencoder runs alongside to flag unfamiliar behaviour as an unknown attack.</p>
<p class="muted">Dataset: CIC IoT-DIAD 2024 (Canadian Institute for Cybersecurity), 49 flow features.</p></div>""",
                unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="card"><h4>Key features</h4><ul style="margin:0; padding-left:1.1rem">
<li><b>Zero-day detection</b>: a benign-only autoencoder flags attack families never seen in training</li>
<li><b>PCAP → verdict</b>: built-in flow extractor for Wireshark captures</li>
<li><b>Plain-English explanations</b> generated from SHAP, with mitigation advice</li>
<li><b>Adversarial robustness</b>: FGSM stress test and adversarial training</li>
<li><b>Lightweight model</b>: SHAP-selected features for edge devices</li>
<li><b>Live monitor</b> and <b>PDF incident reports</b></li></ul></div>""", unsafe_allow_html=True)
