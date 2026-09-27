import streamlit as st

from common import deploy_info, header, kpis, require_analysis

header("Incident report", "Export the current analysis as a PDF: summary, top attackers, most critical flows and "
       "recommended responses.")
A = require_analysis()
res = A["result"]
kpis([("Source", A["name"], None), ("Flows", f"{len(res):,}", None),
      ("Malicious", f"{int((res['Verdict'] != 'Benign').sum()):,}", None)])

with st.form("report"):
    analyst = st.text_input("Analyst name (optional)")
    notes = st.text_area("Notes (optional)", height=110)
    go = st.form_submit_button("Generate PDF", type="primary", icon=":material/picture_as_pdf:")
if go:
    from report_pdf import build_report
    st.session_state["pdf"] = build_report(res, A["name"], analyst, notes, deploy_info())
if "pdf" in st.session_state:
    st.download_button("Download report", st.session_state["pdf"], file_name="ExplainIDS_incident_report.pdf",
                       mime="application/pdf", icon=":material/download:")
