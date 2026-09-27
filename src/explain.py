"""Global and local explainability: SHAP (tree + permutation) and LIME."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from config import FEATURES, FIGURES


def tree_shap_per_class(xgb_model, X, class_names):
    """Exact TreeSHAP on the raw-feature XGBoost -> mean |SHAP| per class (F x C).

    Uses XGBoost's native TreeSHAP (pred_contribs), which is identical to
    shap.TreeExplainer but also works with XGBoost 3.x multi-class models.
    """
    import xgboost as xgb
    contrib = xgb_model.get_booster().predict(xgb.DMatrix(np.asarray(X)), pred_contribs=True)
    contrib = np.asarray(contrib)
    if contrib.ndim == 2:                   # binary: (n, F+1)
        contrib = contrib[:, None, :]
    sv = contrib[:, :, :-1]                 # drop bias column -> (n, C, F)
    return np.abs(sv).mean(axis=0).T        # (F, C)


def attack_score_shap(hybrid, prep_background, X_explain, benign_idx):
    """Permutation SHAP of P(attack) = 1 - P(Benign) for the deployed hybrid model."""
    f = lambda Z: 1.0 - hybrid.predict_proba(np.asarray(Z, dtype=np.float32))[:, benign_idx]
    masker = shap.maskers.Independent(prep_background, max_samples=len(prep_background))
    expl = shap.PermutationExplainer(f, masker, seed=0)
    return expl(X_explain, max_evals=2 * X_explain.shape[1] + 1, silent=True)


def plot_importance_bar(importance: dict, fname="shap_hybrid_bar.png", top=15,
                        title="Global feature importance: mean |SHAP| of the attack score"):
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#3b82f6")
    ax.set_xlabel("mean |SHAP value|"); ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(FIGURES / fname, dpi=150); plt.close(fig)


def plot_global(sv_expl, X_raw_display, fname_prefix="shap_hybrid"):
    sv_expl.feature_names = FEATURES
    sv_expl.data = np.asarray(X_raw_display)
    plot_importance_bar(dict(zip(FEATURES, np.abs(sv_expl.values).mean(0))), f"{fname_prefix}_bar.png")
    plt.figure()
    shap.plots.beeswarm(sv_expl, max_display=15, show=False)
    plt.title("SHAP summary (beeswarm) - attack score")
    plt.tight_layout(); plt.savefig(FIGURES / f"{fname_prefix}_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close("all")
    plt.figure()
    shap.plots.waterfall(sv_expl[0], max_display=12, show=False)
    plt.tight_layout(); plt.savefig(FIGURES / f"{fname_prefix}_waterfall_example.png", dpi=150, bbox_inches="tight"); plt.close("all")


def plot_class_heatmap(imp_fc, class_names, top=15, fname="shap_per_class_heatmap.png"):
    order = np.argsort(-imp_fc.sum(1))[:top]
    M = imp_fc[order]
    M = M / (M.max(axis=0, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(1.1 * len(class_names) + 3, 0.42 * top + 1.5))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(top), [FEATURES[i] for i in order])
    ax.set_title("Which features drive each attack class (normalised mean |SHAP|)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(FIGURES / fname, dpi=150); plt.close(fig)
    return [FEATURES[i] for i in order]


def lime_explainer(train_Z, class_names):
    from lime.lime_tabular import LimeTabularExplainer
    return LimeTabularExplainer(train_Z, feature_names=FEATURES, class_names=class_names,
                                discretize_continuous=True, random_state=0, mode="classification")


def lime_for(explainer, predict_fn, z_row, label_idx, num_features=10):
    exp = explainer.explain_instance(z_row, predict_fn, num_features=num_features, labels=[label_idx],
                                     num_samples=1500)
    return exp.as_list(label=label_idx)
