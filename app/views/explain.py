import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import MODELS, SEV_COLORS, T, colors, chart, header, require_analysis, require_models, section, tag
from config import FEATURES

header("Explain", "Why did the model flag this flow? The page shows SHAP attributions, LIME rules, a plain-English "
       "summary and a recommended response.")
det = require_models()
A = require_analysis()
res = A["result"]

flagged = res.index[res["Verdict"] != "Benign"].tolist()
opts = flagged + [i for i in res.index if i not in set(flagged)]


def _label(i):
    s = f"#{i}  ·  {res.at[i, 'Verdict']}  ·  {res.at[i, 'Confidence']:.0%}"
    if "Src IP" in res.columns:
        s += f"  ·  {res.at[i, 'Src IP']} → {res.at[i, 'Dst IP']}:{res.at[i, 'Dst Port']}"
    return s


sel = st.selectbox("Flow", opts, format_func=_label, help="Attacks are listed first")
pos = res.index.get_loc(sel)
cache = st.session_state.setdefault("_xai_cache", {})
key = (A["name"], int(pos))
if key not in cache:
    with st.spinner("Computing SHAP values…"):
        cache[key] = det.explain(A["Z"][pos], A["X_raw"].iloc[pos].to_numpy(), A["proba"][pos],
                                 res.at[sel, "Verdict"], float(res.at[sel, "Anomaly Score"]))
E = cache[key]
verdict, sev = E["verdict"], res.at[sel, "Severity"]
c = colors().get(verdict, T()["muted"])

st.markdown(f"""<div class="verdict" style="--c:{c}">
<div class="t">{verdict}</div>
<div style="margin:.35rem 0 .6rem 0">{tag('Severity: ' + sev, SEV_COLORS.get(sev, T()['muted']))}
{tag(f"Confidence {res.at[sel, 'Confidence']:.0%}", T()['muted'])}{tag(f"Anomaly {res.at[sel, 'Anomaly Score']:.4f}", T()['muted'])}</div>
<div class="body">{E['text']}</div>
<div class="muted" style="margin-top:.6rem"><b>Recommended response:</b> {E['mitigation']}</div>
</div>""", unsafe_allow_html=True)

t1, t2, t3 = st.tabs(["SHAP attribution", "LIME rules", "Compared with benign traffic"])
sv, raw = np.asarray(E["shap"]), np.asarray(E["raw"])

with t1:
    a, b = st.columns([1.6, 1], gap="large")
    with a:
        order = np.argsort(-np.abs(sv))[:10][::-1]
        fig = go.Figure(go.Bar(x=sv[order], y=[FEATURES[i] for i in order], orientation="h",
                               marker_color=["#ef4444" if v > 0 else T()["accent"] for v in sv[order]],
                               customdata=[raw[i] for i in order], hovertemplate="%{y}<br>value = %{customdata:,.4g}<br>SHAP = %{x:.4f}<extra></extra>"))
        chart(fig, 400, title=f"Top features for P({E['class']})", xaxis_title="SHAP value", bargap=.3)
        st.caption("Red bars push the flow toward this class; blue bars push it away.")
    with b:
        pr = pd.DataFrame({"Class": det.classes, "p": E["proba"]}).sort_values("p")
        fig = go.Figure(go.Bar(x=pr["p"], y=pr["Class"], orientation="h", marker_color=[colors().get(k, T()["muted"]) for k in pr["Class"]],
                               text=[f"{p:.0%}" if p >= .01 else "" for p in pr["p"]], textposition="outside"))
        chart(fig, 400, title="Class probabilities", xaxis=dict(range=[0, 1.15], tickformat=".0%"), bargap=.3)

with t2:
    st.caption("LIME fits a simple local model around this flow and reports human-readable rules.")
    if st.button("Run LIME", icon=":material/play_arrow:"):
        from explain import lime_explainer, lime_for
        if "_lime" not in st.session_state:
            st.session_state["_lime"] = lime_explainer(np.load(MODELS / "lime_train_raw.npy"), det.classes)
        fn = lambda R: det.model.predict_proba(det.prep.transform(R))
        with st.spinner("Sampling…"):
            rules = lime_for(st.session_state["_lime"], fn, raw.astype(np.float64), det.classes.index(E["class"]))
        cache[key]["lime"] = rules
    if "lime" in cache[key]:
        L = pd.DataFrame(cache[key]["lime"], columns=["Rule", "Weight"]).iloc[::-1]
        fig = go.Figure(go.Bar(x=L["Weight"], y=L["Rule"], orientation="h",
                               marker_color=["#ef4444" if w > 0 else T()["accent"] for w in L["Weight"]]))
        chart(fig, 380, title=f"LIME rules for {E['class']}", xaxis_title="weight", bargap=.3)

with t3:
    top = np.argsort(-np.abs(sv))[:10]
    cmp_df = pd.DataFrame({"Feature": [FEATURES[i] for i in top], "This flow": [raw[i] for i in top],
                           "Typical benign (median)": [det.benign_median[FEATURES[i]] for i in top]})
    st.dataframe(cmp_df, width="stretch", hide_index=True,
                 column_config={"This flow": st.column_config.NumberColumn(format="%.4g"),
                                "Typical benign (median)": st.column_config.NumberColumn(format="%.4g")})
