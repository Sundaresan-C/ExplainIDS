import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import T, RESULTS, chart, header, kpis, load_json, require_models, section

header("Zero-day lab", "Can the system catch attacks it has never seen? Each attack family is left out of training in "
       "turn, and the system is then tested on it.")
det = require_models()
zd = load_json("zero_day.json")

if zd:
    z = pd.DataFrame(zd)
    c0, c1 = z["classifier_only_detection"].mean(), z["with_autoencoder_detection"].mean()
    kpis([("Classifier only", f"{c0:.1%}", "avg. detection of unseen family"),
          ("With autoencoder", f"{c1:.1%}", f"{(c1 - c0) * 100:+.1f} points"),
          ("Benign false alarms", f"{z['benign_false_positive_system'].mean():.1%}", "full system")])
    fig = go.Figure()
    fig.add_bar(x=z["held_out"], y=z["classifier_only_detection"], name="Classifier only", marker_color=T()["neutral"])
    fig.add_bar(x=z["held_out"], y=z["with_autoencoder_detection"], name="+ Autoencoder", marker_color=T()["accent"])
    chart(fig, 360, title="Detection rate of the held-out family", barmode="group", bargap=.3,
          yaxis=dict(range=[0, 1.05], tickformat=".0%"), legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    with st.expander("Detailed table"):
        st.dataframe(z, width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in z.columns if c not in ("held_out", "n_test")})
else:
    st.info("Zero-day experiment not run yet. Run `python src/train.py` without `--skip-lofo`.")

section("Threshold explorer", "trade sensitivity against false alarms")


@st.cache_data(show_spinner="Scoring stream…")
def scores():
    S = pd.read_parquet(RESULTS / "live_stream.parquet")
    return det.ae.score(det.prep.transform(S.drop(columns=["Label"]).to_numpy())), S["Label"].to_numpy()


s, y = scores()
pct = st.slider("Threshold (percentile of benign reconstruction error)", 80.0, 99.9, 99.0, 0.1)
thr = float(np.percentile(s[y == "Benign"], pct))
r = pd.DataFrame([{"Class": c, "Flagged": float(np.mean(s[y == c] > thr))} for c in sorted(set(y))])
a, b = st.columns([1.2, 1], gap="medium")
with a:
    fig = go.Figure(go.Bar(x=r["Class"], y=r["Flagged"], marker_color=["#10b981" if c == "Benign" else T()["accent"] for c in r["Class"]],
                           text=[f"{v:.0%}" for v in r["Flagged"]], textposition="outside"))
    chart(fig, 340, title="Share flagged as anomalous", yaxis=dict(range=[0, 1.12], tickformat=".0%"), bargap=.35)
with b:
    lg = np.log10(s + 1e-9)
    fig = go.Figure()
    fig.add_histogram(x=lg[y == "Benign"], name="Benign", marker_color="#10b981", opacity=.7, nbinsx=60)
    fig.add_histogram(x=lg[y != "Benign"], name="Attack", marker_color="#ef4444", opacity=.6, nbinsx=60)
    fig.add_vline(x=np.log10(thr), line_dash="dash", line_color=T()["text"])
    chart(fig, 340, title="Reconstruction error (log10)", barmode="overlay", legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
st.caption(f"Selected threshold {thr:.5f}; deployed threshold {det.ae.threshold:.5f}. The green bar is the benign false-alarm rate.")
