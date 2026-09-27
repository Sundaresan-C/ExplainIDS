"""Headless smoke test of every page of the web app:   python tests/test_app.py"""
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "app"))
from detector import Detector  # noqa: E402

det = Detector()
demo = pd.read_csv(ROOT / "data" / "sample" / "demo_flows_unlabelled.csv")
res, ex = det.analyse(demo, extras=True)
analysis = {"name": "demo_flows_unlabelled.csv", "result": res, **ex}

PAGES = ["home", "detect", "explain", "live", "models", "zero_day", "robustness", "report"]
ok = True
for name in PAGES:
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=240)
    at.session_state["analysis"] = analysis
    at.run()
    if name != "home":
        at.switch_page(f"views/{name}.py").run()
    errs = [e.value for e in at.exception]
    ok &= not errs
    print(f"{'OK ' if not errs else 'ERR'} {name:11s} errors={errs[:1]}")
    if errs:
        continue
    if name == "explain":
        b = [x for x in at.button if x.label == "Run LIME"][0]
        b.click().run()
        print("    LIME:", "ok" if not at.exception else [e.value for e in at.exception][:1])
    if name == "report":
        at.button[0].click().run()
        print("    PDF :", "ok" if len(at.session_state["pdf"]) > 1000 else "missing")
print("ALL PAGES OK" if ok else "SOME PAGES FAILED")
