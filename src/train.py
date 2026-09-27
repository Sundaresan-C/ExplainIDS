"""End-to-end training + evaluation of every model and experiment.

    python src/train.py                 # full run on data/processed/flows.parquet
    python src/train.py --quick         # fast smoke test (fewer epochs / samples)
    python src/train.py --skip-lofo --skip-robust

Outputs:
    models/   deployable artefacts used by the web app
    results/  metrics.json, zero_day.json, robustness.json, lightweight.json, figures/
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import explain
import robustness
from config import CATEGORIES, DATA_PROCESSED, FEATURES, FIGURES, MODELS, RESULTS, SEED
from models import FlowAutoencoder, HybridCNNXGB, build_cnn, dump, fit_cnn, make_xgb
from preprocess import make_splits

RNG = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
def metrics(y, p, proba=None, n_classes=None):
    m = {
        "accuracy": accuracy_score(y, p),
        "precision_macro": precision_score(y, p, average="macro", zero_division=0),
        "recall_macro": recall_score(y, p, average="macro", zero_division=0),
        "f1_macro": f1_score(y, p, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, p, average="weighted", zero_division=0),
    }
    # specificity (macro) = mean over classes of TN / (TN + FP)
    cm = confusion_matrix(y, p, labels=range(n_classes or (max(y.max(), p.max()) + 1)))
    fp = cm.sum(0) - np.diag(cm); tn = cm.sum() - cm.sum(0) - cm.sum(1) + np.diag(cm)
    m["specificity_macro"] = float(np.mean(tn / np.maximum(tn + fp, 1)))
    if proba is not None:
        try:
            m["roc_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr", average="macro")
        except ValueError:
            m["roc_auc_ovr"] = None
    return {k: (round(float(v), 5) if v is not None else None) for k, v in m.items()}


def plot_cm(y, p, names, title, fname):
    cm = confusion_matrix(y, p, labels=range(len(names)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 3, 0.9 * len(names) + 2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.1%}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right"); ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04); fig.tight_layout(); fig.savefig(FIGURES / fname, dpi=150); plt.close(fig)


def plot_history(h, fname):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val"); ax[0].set_title("CNN loss")
    ax[1].plot(h["accuracy"], label="train"); ax[1].plot(h["val_accuracy"], label="val"); ax[1].set_title("CNN accuracy")
    for a in ax:
        a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGURES / fname, dpi=150); plt.close(fig)


def bench(name, fit_fn, predict_proba_fn, S, n_classes, store):
    t = time.perf_counter(); fit_fn(); train_s = time.perf_counter() - t
    t = time.perf_counter(); P = predict_proba_fn(S["Xte"]); inf_s = time.perf_counter() - t
    y = S["enc"].transform(S["yte_str"]); p = P.argmax(1)
    m = metrics(y, p, P, n_classes)
    m.update(train_time_s=round(train_s, 2), inference_ms_per_flow=round(inf_s * 1000 / len(y), 5))
    store[name] = m
    print(f"{name:28s} acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} train={train_s:.1f}s")
    return P


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PROCESSED / "flows.parquet"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-lofo", action="store_true")
    ap.add_argument("--skip-robust", action="store_true")
    ap.add_argument("--shap-samples", type=int, default=300)
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.shap_samples = 5, 80

    df = pd.read_parquet(args.data)
    source = json.loads((DATA_PROCESSED / "source.json").read_text()) if (DATA_PROCESSED / "source.json").exists() else {}
    print(f"dataset: {len(df)} rows; classes:\n{df['Label'].value_counts()}")
    S = make_splits(df)
    enc, prep = S["enc"], S["prep"]
    names = list(enc.classes_); C = len(names); F = len(FEATURES)
    yte = enc.transform(S["yte_str"])
    benign_idx = names.index("Benign")
    results = {}

    # ---------------- baselines ------------------------------------------
    knn_idx = RNG.choice(len(S["Xtr"]), min(len(S["Xtr"]), 40_000), replace=False)
    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
    }
    for name, mdl in base_models.items():
        Xf, yf = (S["Xtr"][knn_idx], S["ytr"][knn_idx]) if name.startswith("KNN") else (S["Xtr"], S["ytr"])
        bench(name, lambda: mdl.fit(Xf, yf), mdl.predict_proba, S, C, results)

    xgb_raw = make_xgb(C)
    bench("XGBoost", lambda: xgb_raw.fit(S["Xtr"], S["ytr"], eval_set=[(S["Xva"], S["yva"])], verbose=False),
          xgb_raw.predict_proba, S, C, results)

    cnn = build_cnn(F, C)
    hist = {}
    def _fit_cnn():
        hist.update(fit_cnn(cnn, S["Xtr"], S["ytr"], S["Xva"], S["yva"], epochs=args.epochs))
    bench("1D-CNN", _fit_cnn, lambda X: cnn.predict(X, batch_size=4096, verbose=0), S, C, results)
    plot_history(hist, "cnn_training_curves.png")

    # ---------------- hybrid: CNN -> XGBoost ------------------------------
    hyb = HybridCNNXGB(F, C, fusion=False, cnn=cnn)
    P_h = bench("Hybrid CNN-XGBoost", lambda: hyb.fit(S["Xtr"], S["ytr"], S["Xva"], S["yva"], train_cnn=False),
                hyb.predict_proba, S, C, results)
    results["Hybrid CNN-XGBoost"]["train_time_s"] += results["1D-CNN"]["train_time_s"]
    # ---------------- ours: CNN embedding (+) raw features -> XGBoost ----
    fus = HybridCNNXGB(F, C, fusion=True, cnn=cnn)
    P_f = bench("Fusion CNN-XGBoost", lambda: fus.fit(S["Xtr"], S["ytr"], S["Xva"], S["yva"], train_cnn=False),
                fus.predict_proba, S, C, results)
    results["Fusion CNN-XGBoost"]["train_time_s"] += results["1D-CNN"]["train_time_s"]

    deploy, deploy_name, P_d = ((fus, "fusion", P_f) if results["Fusion CNN-XGBoost"]["f1_macro"]
                                >= results["Hybrid CNN-XGBoost"]["f1_macro"] else (hyb, "hybrid", P_h))
    print(f"deploying: {deploy_name}")
    plot_cm(yte, P_d.argmax(1), names, f"Confusion matrix - {deploy_name} model", "confusion_matrix_deployed.png")
    plot_cm(yte, P_h.argmax(1), names, "Confusion matrix - hybrid", "confusion_matrix_hybrid.png")
    report = classification_report(yte, P_d.argmax(1), target_names=names, output_dict=True, zero_division=0)

    # binary view (attack vs benign) of the deployed model
    yb, pb = (yte != benign_idx).astype(int), (P_d.argmax(1) != benign_idx).astype(int)
    binary = metrics(yb, pb, None, 2); binary["roc_auc"] = round(float(roc_auc_score(yb, 1 - P_d[:, benign_idx])), 5)

    # comparison chart
    order = list(results)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    xs = np.arange(len(order)); w = 0.38
    ax.bar(xs - w / 2, [results[k]["accuracy"] for k in order], w, label="Accuracy")
    ax.bar(xs + w / 2, [results[k]["f1_macro"] for k in order], w, label="Macro F1")
    ax.set_xticks(xs, order, rotation=25, ha="right"); ax.set_ylim(max(0, min(results[k]["f1_macro"] for k in order) - 0.05), 1.0)
    ax.legend(); ax.grid(axis="y", alpha=.3); ax.set_title("Model comparison (test set)")
    fig.tight_layout(); fig.savefig(FIGURES / "model_comparison.png", dpi=150); plt.close(fig)

    # ---------------- persist deployable artefacts -----------------------
    deploy.save(MODELS, deploy_name)
    hyb.save(MODELS, "hybrid") if deploy_name != "hybrid" else None
    dump(prep, MODELS / "preprocessor.joblib"); dump(enc, MODELS / "label_encoder.joblib")
    xgb_raw.save_model(MODELS / "xgb_raw.json")
    bg = S["Xtr"][RNG.choice(len(S["Xtr"]), 50, replace=False)]
    np.save(MODELS / "shap_background.npy", bg)
    benign_raw = S["Xtr_raw_benign"]
    (MODELS / "benign_median.json").write_text(json.dumps({f: float(benign_raw[f].median()) for f in FEATURES}))
    lime_idx = RNG.choice(len(S["Xtr"]), min(len(S["Xtr"]), 5000), replace=False)
    np.save(MODELS / "lime_train_raw.npy", prep.inverse_transform(S["Xtr"][lime_idx]).astype(np.float32))

    # ---------------- zero-day autoencoder --------------------------------
    Zb_tr = S["Xtr"][S["ytr"] == benign_idx]; Zb_va = S["Xva"][S["yva"] == benign_idx]
    ae = FlowAutoencoder(F).fit(Zb_tr, Zb_va, epochs=15 if args.quick else 60)
    ae.save(MODELS)
    ae_te = ae.score(S["Xte"])
    ae_auc = float(roc_auc_score(yb, ae_te))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(np.log10(ae_te[yb == 0] + 1e-9), 80, alpha=.6, label="Benign", density=True)
    ax.hist(np.log10(ae_te[yb == 1] + 1e-9), 80, alpha=.6, label="Attacks", density=True)
    ax.axvline(np.log10(ae.threshold), color="red", ls="--", label="threshold (99th pct benign)")
    ax.set_xlabel("log10 reconstruction error"); ax.legend(); ax.set_title(f"Autoencoder anomaly scores (AUC={ae_auc:.3f})")
    fig.tight_layout(); fig.savefig(FIGURES / "autoencoder_scores.png", dpi=150); plt.close(fig)

    # ---------------- explainability -------------------------------------
    print("computing SHAP ...")
    k = min(args.shap_samples, len(S["Xte"]))
    idx = RNG.choice(len(S["Xte"]), k, replace=False)
    sv = explain.attack_score_shap(deploy, bg, S["Xte"][idx], benign_idx)
    explain.plot_global(sv, S["Xte_raw"].iloc[idx].to_numpy())
    global_imp = dict(zip(FEATURES, np.abs(sv.values).mean(0).round(6).tolist()))
    tree_n = min(len(S["Xte"]), 4000)
    imp_fc = explain.tree_shap_per_class(xgb_raw, S["Xte"][:tree_n], names)
    top_feats = explain.plot_class_heatmap(imp_fc, names)
    per_class_imp = {names[c]: dict(zip(FEATURES, imp_fc[:, c].round(6).tolist())) for c in range(C)}

    # ---------------- lightweight models (SHAP-ranked top-k) --------------
    print("lightweight study ...")
    rank = np.argsort(-imp_fc.sum(1))
    light = []
    for kf in [5, 10, 15, 20, 30, F]:
        cols = rank[:kf]
        m = make_xgb(C)
        t = time.perf_counter(); m.fit(S["Xtr"][:, cols], S["ytr"], eval_set=[(S["Xva"][:, cols], S["yva"])], verbose=False)
        tr = time.perf_counter() - t
        t = time.perf_counter(); p = m.predict(S["Xte"][:, cols]); inf = time.perf_counter() - t
        path = MODELS / f"_tmp_light_{kf}.json"; m.save_model(path); size = os.path.getsize(path); path.unlink()
        light.append({"k": int(kf), "features": [FEATURES[i] for i in cols], "accuracy": float(accuracy_score(yte, p)),
                      "f1_macro": float(f1_score(yte, p, average="macro")), "model_kb": round(size / 1024, 1),
                      "train_s": round(tr, 2), "inference_us_per_flow": round(inf * 1e6 / len(yte), 3)})
        print(f"  top-{kf:2d}: f1={light[-1]['f1_macro']:.4f} size={light[-1]['model_kb']}KB")
    (RESULTS / "lightweight.json").write_text(json.dumps(light, indent=2))

    # ---------------- zero-day: leave-one-family-out ----------------------
    if not args.skip_lofo:
        print("zero-day leave-one-family-out ...")
        zd = []
        for fam in [c for c in names if c != "Benign"]:
            Sx = make_splits(df, exclude=[fam])
            ncls = len(Sx["enc"].classes_); bidx = list(Sx["enc"].classes_).index("Benign")
            clf = make_xgb(ncls).fit(Sx["Xtr"], Sx["ytr"], eval_set=[(Sx["Xva"], Sx["yva"])], verbose=False)
            aex = FlowAutoencoder(F).fit(Sx["Xtr"][Sx["ytr"] == bidx], Sx["Xva"][Sx["yva"] == bidx],
                                         epochs=10 if args.quick else 40, verbose=0)
            P = clf.predict_proba(Sx["Xte"]); pred = P.argmax(1); conf = P.max(1)
            s = aex.score(Sx["Xte"]); anom = s > aex.threshold
            unknown = anom & ((pred == bidx) | (conf < 0.6))
            is_fam = Sx["yte_str"] == fam; is_ben = Sx["yte_str"] == "Benign"
            clf_detect = float(np.mean(pred[is_fam] != bidx))
            sys_detect = float(np.mean((pred[is_fam] != bidx) | unknown[is_fam]))
            zd.append({"held_out": fam, "n_test": int(is_fam.sum()),
                       "classifier_only_detection": round(clf_detect, 4),
                       "with_autoencoder_detection": round(sys_detect, 4),
                       "flagged_unknown_rate": round(float(np.mean(unknown[is_fam])), 4),
                       "benign_false_positive_classifier": round(float(np.mean(pred[is_ben] != bidx)), 4),
                       "benign_false_positive_system": round(float(np.mean((pred[is_ben] != bidx) | unknown[is_ben])), 4)})
            print(f"  {fam:11s} classifier={clf_detect:.3f} -> with AE={sys_detect:.3f}")
        (RESULTS / "zero_day.json").write_text(json.dumps(zd, indent=2))
        fig, ax = plt.subplots(figsize=(9, 3.8))
        xs = np.arange(len(zd)); w = .38
        ax.bar(xs - w / 2, [z["classifier_only_detection"] for z in zd], w, label="Supervised classifier only")
        ax.bar(xs + w / 2, [z["with_autoencoder_detection"] for z in zd], w, label="+ Autoencoder")
        ax.set_xticks(xs, [z["held_out"] for z in zd]); ax.set_ylim(0, 1.05); ax.set_ylabel("Detection rate of unseen family")
        ax.legend(); ax.grid(axis="y", alpha=.3); ax.set_title("Zero-day detection: attack family never seen in training")
        fig.tight_layout(); fig.savefig(FIGURES / "zero_day_detection.png", dpi=150); plt.close(fig)

    # ---------------- adversarial robustness ------------------------------
    if not args.skip_robust:
        print("robustness ...")
        n = min(len(S["Xte"]), 8000)
        sub = RNG.choice(len(S["Xte"]), n, replace=False)
        std_rows = robustness.evaluate(deploy, S["Xte"][sub], yte[sub])
        robust = robustness.adversarial_training(deploy, S["Xtr"], S["ytr"], S["Xva"], S["yva"],
                                                 epochs=2 if args.quick else 6)
        rob_rows = robustness.evaluate(robust, S["Xte"][sub], yte[sub])
        for r in std_rows: r["model"] = "Standard"
        for r in rob_rows: r["model"] = "Adversarially trained"
        robust.save(MODELS, "robust")
        (RESULTS / "robustness.json").write_text(json.dumps(std_rows + rob_rows, indent=2))
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for rows, st in ((std_rows, "-"), (rob_rows, "--")):
            for kind, mk in (("FGSM", "o"), ("Gaussian noise", "s")):
                r = [x for x in rows if x["attack"] == kind]
                ax.plot([x["epsilon"] for x in r], [x["accuracy"] for x in r], st, marker=mk, label=f"{r[0]['model']} - {kind}")
        ax.set_xlabel("perturbation ε (scaled feature space)"); ax.set_ylabel("accuracy"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title("Robustness to adversarial / noisy inputs")
        fig.tight_layout(); fig.savefig(FIGURES / "robustness.png", dpi=150); plt.close(fig)

    # ---------------- live-stream replay file + summary -------------------
    live = S["Xte_raw"].copy(); live["Label"] = S["yte_str"]
    live.sample(min(len(live), 5000), random_state=SEED).to_parquet(RESULTS / "live_stream.parquet", index=False)

    (MODELS / "deploy.json").write_text(json.dumps({
        "model_name": deploy_name, "classes": names, "features": FEATURES, "source": source.get("source", "unknown"),
        "trained_at": time.strftime("%Y-%m-%d %H:%M"), "ae_threshold": ae.threshold}, indent=2))
    summary = {
        "dataset": {"source": source.get("source", "unknown"), "rows": int(len(df)),
                    "class_counts": df["Label"].value_counts().to_dict(),
                    "train_rows_after_smote": int(len(S["Xtr"])), "test_rows": int(len(S["Xte"]))},
        "models": results, "deployed": deploy_name, "binary_attack_vs_benign": binary,
        "per_class_report": report, "autoencoder": {"threshold": ae.threshold, "auc_attack_vs_benign": round(ae_auc, 5)},
        "global_shap_attack_score": global_imp, "per_class_tree_shap": per_class_imp, "top_features": top_feats,
        "cnn_history": {k: [float(x) for x in v] for k, v in hist.items()},
    }
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("done. results ->", RESULTS)


if __name__ == "__main__":
    main()
