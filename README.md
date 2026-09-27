# ExplainIDS: an explainable, zero-day-aware network intrusion detection system

A final-year project based on the IEEE Access 2025 paper:

> P. Jain, A. Rathour, A. Sharma, G. S. Chhabra, **"Bridging Explainability and Security: An XAI-Enhanced Hybrid Deep
> Learning Framework for IoT Device Identification and Attack Detection,"** *IEEE Access*, vol. 13, pp. 127368–127390,
> 2025. DOI: [10.1109/ACCESS.2025.3590159](https://doi.org/10.1109/ACCESS.2025.3590159) (open access, CC BY 4.0)

No physical IoT hardware is needed. The project trains on the public **CIC IoT-DIAD 2024** dataset and can analyse any
`.pcap` capture or CICFlowMeter CSV.

## What is reproduced from the paper
* A hybrid **1D-CNN → XGBoost** classifier, where the CNN learns deep features and XGBoost classifies them.
* **SHAP** and **LIME** explainability.
* A comparison against classical ML baselines: LR, DT, KNN, RF, XGBoost and a plain CNN.

## What is new, beyond the paper
| # | Extension | Where |
|---|-----------|-------|
| 1 | **Zero-day detection**: an autoencoder trained on benign traffic only, an open-set decision rule, and a leave-one-attack-family-out evaluation | `src/models.py`, `src/train.py`, *Zero-Day Lab* page |
| 2 | **PCAP → prediction**: our own CICFlowMeter-style flow extractor | `src/flow_extractor.py` |
| 3 | **Plain-English explanations** built from SHAP, plus mitigation advice | `src/nl_explain.py` |
| 4 | **Adversarial robustness**: FGSM and noise stress tests, plus FGSM adversarial training | `src/robustness.py` |
| 5 | **Lightweight model**: SHAP-ranked top-k features for edge devices | `src/train.py` |
| 6 | **Feature-fusion hybrid**: CNN embedding ⊕ raw features → XGBoost | `src/models.py` |
| 7 | **Leakage-aware features**: no IPs, ports or timestamps | `src/config.py` |
| 8 | **Web app**: live SOC monitor and PDF incident report | `app/` |

## Setup (Windows)
Install Python 3.11 (tick "Add to PATH"), then double-click **`setup.bat`**. Or run the steps manually:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Get the dataset
1. Open https://www.unb.ca/cic/datasets/iot-diad-2024.html, fill in the form, and open the download folder.
2. From **`Dataset/AD_Flow-based-features/`**, download CSVs for Benign and for every attack category: DDoS, DoS, Recon,
   Web-based, BruteForce, Spoofing and Mirai. Keep the folder names, because labels are inferred from them.
3. Put them under `data/raw/`, for example `data/raw/DDoS/DDoS-SYN_Flood/....csv`.

## Run
```bash
python src/data_loader.py --per-class 25000   # build data/processed/flows.parquet
python src/train.py                           # trains everything and writes results/ and models/ (~20-40 min on CPU)
streamlit run app/streamlit_app.py            # opens the web app at http://localhost:8501
```
To smoke-test before downloading the real dataset:
```bash
python src/make_sample_data.py                # synthetic PCAPs + CSV into data/sample/
python src/data_loader.py --source sample
python src/train.py --quick
```
Results from synthetic data are for testing only. Use CIC IoT-DIAD 2024 for the report.

On Windows the whole pipeline is one click: **`run_pipeline.bat`** (data → training → report/slides), then **`run_app.bat`**.

## Demo tips
* `http://localhost:8501/detect?demo=1` opens with the demo capture already analysed.
* `http://localhost:8501/live?autoplay=1` starts the live monitor automatically.
* With the app running, `python docs/capture_ui.py` refreshes the UI screenshots used in the report and slides.
  Then re-run `python docs/build_report.py` and `python docs/build_slides.py`.
* `python tests/test_app.py` runs a headless smoke test of every page.

## Project structure
```
src/   config, data_loader, flow_extractor, preprocess, models, explain, robustness,
       nl_explain, detector, report_pdf, train, make_sample_data
app/   streamlit_app.py + views/ (Home, Detect, Explain, Live monitor, Model comparison,
       Zero-day lab, Robustness, Incident report)
docs/  Project_Report.docx, Project_Presentation.pptx, base_paper/
```
"# ExplainIDS" 
