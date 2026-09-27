import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import RESULTS, colors, chart, header, kpis, require_models, section

header("Live monitor", "Held-out test flows are replayed as a real-time stream and classified as they arrive.")
det = require_models()
path = RESULTS / "live_stream.parquet"
if not path.exists():
    st.error("results/live_stream.parquet is missing. Run training first.")
    st.stop()


@st.cache_data
def stream():
    return pd.read_parquet(path).sample(frac=1.0, random_state=7).reset_index(drop=True)


S = stream()
ss = st.session_state
ss.setdefault("live_pos", 0); ss.setdefault("live_log", []); ss.setdefault("live_running", False)
if "autoplay" in st.query_params and not ss.get("_autoplayed"):
    ss.live_running = ss._autoplayed = True

c1, c2, c3, c4 = st.columns([1, 1, 1, 3], vertical_alignment="bottom")
if c1.button("Start", icon=":material/play_arrow:", type="primary", width="stretch", disabled=ss.live_running):
    ss.live_running = True; st.rerun()
if c2.button("Pause", icon=":material/pause:", width="stretch", disabled=not ss.live_running):
    ss.live_running = False; st.rerun()
if c3.button("Reset", icon=":material/restart_alt:", width="stretch"):
    ss.live_pos, ss.live_log, ss.live_running = 0, [], False; st.rerun()
rate = c4.slider("Flows per second", 5, 200, 40)


def step():
    if not ss.live_running or ss.live_pos >= len(S):
        if ss.live_pos >= len(S):
            ss.live_running = False
        return
    batch = S.iloc[ss.live_pos: ss.live_pos + rate]
    r = det.analyse(batch.drop(columns=["Label"]))
    tick = len({e["tick"] for e in ss.live_log})
    ss.live_log.extend({"tick": tick, "Verdict": v, "Severity": s, "Confidence": float(cf), "Truth": t}
                       for v, s, cf, t in zip(r["Verdict"], r["Severity"], r["Confidence"], batch["Label"]))
    ss.live_log = ss.live_log[-20000:]
    ss.live_pos += len(batch)


def render():
    log = pd.DataFrame(ss.live_log)
    if log.empty:
        kpis([("Flows", "–", None), ("Attacks", "–", None), ("Unknown", "–", None), ("Accuracy", "–", None), ("Top threat", "–", None)])
        st.caption("Press Start to begin streaming.")
        return
    n = len(log); att = int((log["Verdict"] != "Benign").sum())
    known = log[log["Verdict"] != "Unknown attack"]
    acc = (known["Verdict"] == known["Truth"]).mean() if len(known) else np.nan
    threats = log.loc[log["Verdict"] != "Benign", "Verdict"]
    kpis([("Flows", f"{n:,}", f"{ss.live_pos / len(S):.0%} of stream"), ("Attacks", f"{att:,}", f"{att / n:.0%} of traffic"),
          ("Unknown", f"{int((log['Verdict'] == 'Unknown attack').sum()):,}", "zero-day"),
          ("Accuracy", f"{acc:.1%}" if acc == acc else "–", "vs. ground truth"),
          ("Top threat", threats.mode().iloc[0] if len(threats) else "None", None)])
    a, b = st.columns([1.7, 1], gap="medium")
    with a:
        per = log.groupby(["tick", "Verdict"]).size().reset_index(name="Flows")
        fig = px.bar(per, x="tick", y="Flows", color="Verdict", color_discrete_map=colors())
        chart(fig, 360, title="Flows per second", xaxis_title="time (s)", legend_title=None, bargap=.15)
    with b:
        section("Alert feed", "latest first")
        alerts = log[log["Verdict"] != "Benign"].tail(50).iloc[::-1]
        st.dataframe(alerts[["tick", "Verdict", "Severity", "Confidence"]], width="stretch", hide_index=True, height=300,
                     column_config={"tick": st.column_config.NumberColumn("t (s)"),
                                    "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})


@st.fragment(run_every=1.0 if ss.live_running else None)
def live():
    step()
    render()
    if not ss.live_running and ss.live_pos >= len(S):
        st.success("Stream finished. Press Reset to replay.")


live()
