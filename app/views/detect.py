import io
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from common import T, ROOT, colors, chart, header, kpis, load_demo_if_requested, require_models, section

header("Detect", "Upload a packet capture (.pcap / .pcapng) or a CICFlowMeter CSV. Each flow is classified, checked for "
       "zero-day behaviour and given a severity.")
det = require_models()
SAMPLE = ROOT / "data" / "sample"


@st.cache_data(show_spinner="Extracting flows from capture…")
def flows_from_pcap(data: bytes, suffix: str):
    from flow_extractor import extract_flows
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name
    try:
        return extract_flows(path)
    finally:
        Path(path).unlink(missing_ok=True)


with st.container(border=True):
    up = st.file_uploader("Traffic file", type=["pcap", "pcapng", "cap", "csv"], label_visibility="collapsed")
    with st.container(horizontal=True, gap="small"):
        demo_pcap = st.button("Demo PCAP", icon=":material/play_arrow:", width="content",
                              disabled=not (SAMPLE / "demo_mixed_traffic.pcap").exists())
        demo_csv = st.button("Demo CSV", icon=":material/table_view:", width="content",
                             disabled=not (SAMPLE / "demo_flows_unlabelled.csv").exists())

src_df, src_name = None, None
if up is not None and st.session_state.get("_last_upload") != (up.name, up.size):
    raw = up.getvalue()
    if up.name.lower().endswith(".csv"):
        src_df = pd.read_csv(io.BytesIO(raw), low_memory=False)
        src_df.columns = [c.strip() for c in src_df.columns]
    else:
        src_df = flows_from_pcap(raw, Path(up.name).suffix)
    src_name = up.name
    st.session_state["_last_upload"] = (up.name, up.size)
elif demo_pcap:
    src_df, src_name = flows_from_pcap((SAMPLE / "demo_mixed_traffic.pcap").read_bytes(), ".pcap"), "demo_mixed_traffic.pcap"
elif demo_csv:
    src_df, src_name = pd.read_csv(SAMPLE / "demo_flows_unlabelled.csv"), "demo_flows_unlabelled.csv"

if src_df is not None:
    if src_df.empty:
        st.error("No IP flows found in this file.")
        st.stop()
    with st.spinner(f"Analysing {len(src_df):,} flows…"):
        res, ex = det.analyse(src_df, extras=True)
    st.session_state["analysis"] = {"name": src_name, "result": res, **ex}
    st.session_state.pop("_xai_cache", None); st.session_state.pop("pdf", None)

load_demo_if_requested()
A = st.session_state.get("analysis")
if not A:
    st.caption("Upload a file, or try one of the demo files.")
    st.stop()

res = A["result"]
n = len(res); att = int((res["Verdict"] != "Benign").sum()); unk = int((res["Verdict"] == "Unknown attack").sum())
section("Results", A["name"])
kpis([
    ("Flows", f"{n:,}", None),
    ("Malicious", f"{att:,}", f"{att / n:.1%} of flows"),
    ("Unknown", f"{unk:,}", "zero-day"),
    ("Critical", f"{int((res['Severity'] == 'Critical').sum()):,}", "severity"),
    ("Latency", f"{A['latency_ms'] * 1000:.0f} µs", "per flow"),
])

c1, c2 = st.columns([1, 1.35], gap="medium")
with c1:
    vc = res["Verdict"].value_counts().reset_index(); vc.columns = ["Verdict", "Flows"]
    fig = px.pie(vc, names="Verdict", values="Flows", hole=.62, color="Verdict", color_discrete_map=colors())
    fig.update_traces(textposition="inside", textinfo="percent", sort=False,
                      marker=dict(line=dict(color=T()["bg"], width=2)))
    chart(fig, 320, title="Traffic breakdown", uniformtext=dict(minsize=11, mode="hide"),
          legend=dict(orientation="v", x=1, y=.5))
with c2:
    bad = res[res["Verdict"] != "Benign"]
    if "Src IP" in res.columns and len(bad):
        top = bad.groupby("Src IP").agg(Flows=("Verdict", "size"), Verdict=("Verdict", lambda s: s.mode().iloc[0]))
        top = top.sort_values("Flows").tail(8).reset_index()
        fig = px.bar(top, x="Flows", y="Src IP", color="Verdict", orientation="h", color_discrete_map=colors())
        chart(fig, 320, title="Top attacking sources", yaxis_title=None, legend_title=None, bargap=.35,
              legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"), margin=dict(t=60))
    else:
        sev = res["Severity"].value_counts().reindex(["Critical", "High", "Medium", "Low", "None"]).dropna().reset_index()
        sev.columns = ["Severity", "Flows"]
        fig = px.bar(sev, x="Severity", y="Flows")
        fig.update_traces(marker_color=T()["accent"])
        chart(fig, 320, title="Flows by severity", xaxis_title=None, bargap=.45)

section("Flows")
f1, f2 = st.columns([3, 1])
choices = ["All", "Malicious only", "Unknown only"]
mode = f1.segmented_control("Show", choices, default="Malicious only", label_visibility="collapsed")
view = res if mode in (None, "All") else res[res["Verdict"] != "Benign"] if mode == "Malicious only" else res[res["Verdict"] == "Unknown attack"]
f2.download_button("Export CSV", res.to_csv(index=True).encode(), file_name="explainids_results.csv", mime="text/csv",
                   icon=":material/download:", width="stretch")

cols = [c for c in ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Packets", "Verdict", "Severity", "Confidence",
                    "Anomaly Score"] if c in view.columns]
st.dataframe(
    view[cols].style.map(lambda v, C=colors(): f"color:{C[v]}; font-weight:600" if v in C else "", subset=["Verdict"]),
    width="stretch", height=380,
    column_config={
        "Confidence": st.column_config.ProgressColumn("Confidence", format="percent", min_value=0, max_value=1),
        "Anomaly Score": st.column_config.NumberColumn("Anomaly", format="%.4f"),
        "Src Port": st.column_config.NumberColumn(format="%d"), "Dst Port": st.column_config.NumberColumn(format="%d"),
    })
st.caption(f"{len(view):,} of {n:,} flows shown. To see why a flow was flagged, open the Explain page.")
