import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import T, chart, header, kpis, load_json, section

header("Robustness", "How the detector holds up when an attacker perturbs traffic features, and how far the model can be "
       "shrunk for edge devices.")

rb = load_json("robustness.json")
section("Adversarial robustness", "FGSM and Gaussian noise, ε in scaled feature space")
if rb:
    r = pd.DataFrame(rb)
    eps = st.select_slider("Perturbation strength ε", sorted(r["epsilon"].unique()), value=0.05)
    g = lambda mdl, atk: r[(r.model == mdl) & (r.attack == atk) & (r.epsilon == eps)]["accuracy"].iloc[0]
    kpis([("Standard · FGSM", f"{g('Standard', 'FGSM'):.1%}", None),
          ("Adv. trained · FGSM", f"{g('Adversarially trained', 'FGSM'):.1%}",
           f"{(g('Adversarially trained', 'FGSM') - g('Standard', 'FGSM')) * 100:+.1f} points"),
          ("Standard · noise", f"{g('Standard', 'Gaussian noise'):.1%}", None),
          ("Adv. trained · noise", f"{g('Adversarially trained', 'Gaussian noise'):.1%}", None)])
    fig = go.Figure()
    for mdl, col in (("Standard", T()["neutral"]), ("Adversarially trained", T()["accent"])):
        for atk, dash in (("FGSM", "solid"), ("Gaussian noise", "dot")):
            d = r[(r.model == mdl) & (r.attack == atk)]
            fig.add_scatter(x=d["epsilon"], y=d["accuracy"], mode="lines+markers", name=f"{mdl.split(' (')[0]} · {atk}",
                            line=dict(color=col, dash=dash, width=2))
    fig.add_vline(x=eps, line_color=T()["muted"], opacity=.5)
    chart(fig, 380, title="Accuracy vs. perturbation", xaxis_title="ε", yaxis=dict(tickformat=".0%"),
          legend=dict(orientation="h", y=-0.2))
    st.caption("FGSM is a white-box gradient attack on the CNN feature learner. The adversarially trained model is fine-tuned "
               "on a mix of clean and FGSM examples.")
else:
    st.info("Robustness experiment not run yet.")

section("Lightweight model", "features selected by SHAP ranking")
lw = load_json("lightweight.json")
if lw:
    l = pd.DataFrame(lw)
    best = l[l["f1_macro"] >= l["f1_macro"].max() - 0.005].sort_values("k").iloc[0]
    full = l.sort_values("k").iloc[-1]
    kpis([("Features needed", f"{int(best['k'])} / {int(full['k'])}", "within 0.5 pt of best F1"),
          ("Macro F1", f"{best['f1_macro']:.2%}", f"full model {full['f1_macro']:.2%}"),
          ("Model size", f"{best['model_kb']:.0f} KB", None),
          ("Inference", f"{best['inference_us_per_flow']:.1f} µs", "per flow")])
    fig = go.Figure(go.Scatter(x=l["k"], y=l["f1_macro"], mode="lines+markers", line=dict(color=T()["accent"], width=2)))
    chart(fig, 300, title="Macro F1 vs. number of features", xaxis_title="top-k features", yaxis=dict(tickformat=".1%"))
    with st.expander(f"The {int(best['k'])} selected features"):
        st.write(", ".join(best["features"]))
else:
    st.info("Lightweight study not run yet.")
