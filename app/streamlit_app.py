"""ExplainIDS web app entry point:  streamlit run app/streamlit_app.py"""
from pathlib import Path

import streamlit as st

from common import setup, sidebar_status, theme, theme_toggle

setup()

nav = st.navigation({
    "Overview": [st.Page("views/home.py", title="Home", icon=":material/space_dashboard:", default=True)],
    "Analyse": [
        st.Page("views/detect.py", title="Detect", icon=":material/upload_file:"),
        st.Page("views/explain.py", title="Explain", icon=":material/psychology:"),
        st.Page("views/live.py", title="Live monitor", icon=":material/monitoring:"),
    ],
    "Evaluation": [
        st.Page("views/models.py", title="Model comparison", icon=":material/leaderboard:"),
        st.Page("views/zero_day.py", title="Zero-day lab", icon=":material/science:"),
        st.Page("views/robustness.py", title="Robustness", icon=":material/verified_user:"),
    ],
    "Export": [st.Page("views/report.py", title="Incident report", icon=":material/description:")],
})
st.logo(str(Path(__file__).parent / "assets" / ("logo.svg" if theme() == "dark" else "logo_light.svg")), size="large")
theme_toggle()
sidebar_status()
nav.run()
